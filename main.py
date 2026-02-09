import os
import sqlite3
import pandas as pd
from fastapi import FastAPI
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent
from sqlalchemy import create_engine
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
load_dotenv() 

app = FastAPI()

# 스케줄러 초기화
scheduler = BackgroundScheduler()

@app.on_event("startup")
async def startup_event():
    # 매주 일요일 밤 12시(00:00)에 실행
    scheduler.add_job(
        refresh_local_db,
        trigger=CronTrigger(day_of_week='sun', hour=0, minute=0),
        id='refresh_db_weekly',
        name='Refresh local DB every Sunday at midnight',
        replace_existing=True
    )
    scheduler.start()
    print("스케줄러 시작: 매주 일요일 밤 12시에 DB 동기화가 실행됩니다.")

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()
    print("스케줄러 종료")

# 1. 환경 설정
DB_PATH = "./attendance.db"
# 구글 AI 스튜디오에서 발급받은 API 키를 넣으세요
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def get_agent():
    if not os.path.exists(DB_PATH):
        return None
    
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # 1. 테이블에 대한 상세 설명을 작성합니다.
    custom_prefix = """
        너는 교회 출석 데이터를 분석하는 전문가야. 아래의 테이블 구조와 규칙을 반드시 지켜서 SQL을 생성해줘:
        
        교회 출석 관리 시스템이므로 매주 일요일만 체크하면 된다는 점을 명심해!

        ## 테이블 구조

        ### 1. `student` 테이블
        학생의 기본 정보를 저장하는 테이블이야.

        **주요 컬럼:**
        - `id` (BIGINT, PRIMARY KEY): 학생 고유 ID
        - `name` (VARCHAR, NOT NULL): 학생 이름
        - `birth` (DATE, NOT NULL): 생년월일
        - `sex` (ENUM: 'MAN', 'WOMAN'): 성별
        - `phone` (VARCHAR): 학생 연락처
        - `parent_phone` (VARCHAR): 학부모 연락처
        - `school` (VARCHAR): 다니는 학교
        - `is_graduated` (BOOLEAN, NOT NULL, DEFAULT false): 졸업 여부
        - `deleted_at` (TIMESTAMP): 삭제 시각 (Soft Delete)
        - `created_at`, `updated_at`: 생성/수정 시각

        **관계:**
        - `student_class` 테이블과 1:N 관계 (한 학생은 여러 학년도에 반 배정 가능)

        ---

        ### 2. `student_class` 테이블
        학생과 반(ClassRoom)의 매핑 정보를 저장하는 중간 테이블이야. 학년도별로 학생이 어느 반에 속하는지 관리해.

        **주요 컬럼:**
        - `id` (BIGINT, PRIMARY KEY): 학생-반 매핑 고유 ID
        - `student_id` (BIGINT, FOREIGN KEY, NOT NULL): `student.id` 참조
        - `class_room_id` (BIGINT, FOREIGN KEY, NOT NULL): `class_room.id` 참조
        - `school_year` (INTEGER, NOT NULL): 학년도 (예: 2025)
        - `created_at`, `updated_at`: 생성/수정 시각

        **관계:**
        - `student` 테이블과 N:1 관계 (여러 student_class가 하나의 student를 참조)
        - `class_room` 테이블과 N:1 관계 (여러 student_class가 하나의 class_room을 참조)
        - `attendance` 테이블과 1:N 관계 (하나의 student_class는 여러 출석 기록을 가짐)

        **중요:** 학생 정보를 조회할 때는 반드시 `student_class`를 통해 `student`와 JOIN 해야 해!

        ---

        ### 3. `attendance` 테이블
        출석 기록을 저장하는 테이블이야. 특정 날짜에 특정 학생(student_class)의 출석 상태를 기록해.

        **주요 컬럼:**
        - `id` (BIGINT, PRIMARY KEY): 출석 기록 고유 ID
        - `student_class_id` (BIGINT, FOREIGN KEY, NOT NULL): `student_class.id` 참조
        - `date` (DATE, NOT NULL): 출석한 날짜
        - `status` (ENUM, NOT NULL): 출석 상태
        - `ATTEND`: 출석
        - `LATE`: 지각
        - `ABSENT`: 결석
        - `OTHER`: 기타
        - `UNCHECKED`: 미확인
        - `created_at`, `updated_at`: 생성/수정 시각

        **제약 조건:**
        - `student_class_id`와 `date`의 조합은 UNIQUE (같은 날짜에 같은 학생의 출석 기록은 1개만 존재)

        **관계:**
        - `student_class` 테이블과 N:1 관계 (여러 출석 기록이 하나의 student_class를 참조)

        ---

        ## 테이블 간 관계도

        ```
        student (1) ─────< (N) student_class (1) ─────< (N) attendance
        id                    student_id                  student_class_id
                                class_room_id               date
                                school_year                 status
        ```

        **조인 예시:**
        ```sql
        -- 학생 이름으로 출석 기록 조회
        SELECT s.name, a.date, a.status
        FROM attendance a
        JOIN student_class sc ON a.student_class_id = sc.id
        JOIN student s ON sc.student_id = s.id
        WHERE s.name = '김철수';
        ```

        ---

        ## 중요 규칙

        ### 1. 결석 처리 규칙 ⚠️
        **결석은 2가지 경우가 있어:**

        1. **명시적 결석**: `attendance` 테이블에 `status = 'ABSENT'`로 기록된 경우
        2. **암묵적 결석**: 특정 날짜에 `attendance` 테이블에 아예 기록이 없는 경우 (출석 체크를 하지 않음)

        **예시:**
        - "이번 달 결석한 학생"을 찾을 때는:
        - `status = 'ABSENT'`인 학생 **+**
        - 해당 기간에 `attendance` 테이블에 기록이 전혀 없는 학생 (LEFT JOIN 활용)

        ```sql
        -- 잘못된 예시 (명시적 결석만 조회)
        SELECT s.name FROM attendance a
        JOIN student_class sc ON a.student_class_id = sc.id
        JOIN student s ON sc.student_id = s.id
        WHERE a.status = 'ABSENT';

        -- 올바른 예시 (암묵적 결석도 포함)
        SELECT s.name FROM student s
        JOIN student_class sc ON s.id = sc.student_id
        LEFT JOIN attendance a ON sc.id = a.student_class_id 
        AND a.date BETWEEN '2025-01-01' AND '2025-01-31'
        WHERE a.id IS NULL OR a.status = 'ABSENT';
        ```

        ### 2. 출석 조회 규칙
        - **출석**: `status = 'ATTEND'`
        - **지각**: `status = 'LATE'`
        - **출석으로 인정**: `status IN ('ATTEND', 'LATE')` (지각도 출석으로 간주)

        ### 3. 학생 정보 조회 시 주의사항
        - 학생 이름, 반 정보를 함께 조회할 때는 반드시 `student_class`를 경유해야 해
        - 현재 학년도 학생만 조회하려면 `student_class.school_year = YEAR(CURDATE())` 조건 추가
        - 삭제된 학생은 제외: `student.deleted_at IS NULL`

        ### 4. 날짜 처리
        - "이번 달": `MONTH(a.date) = MONTH(CURDATE()) AND YEAR(a.date) = YEAR(CURDATE())`
        - "지난 주": `a.date BETWEEN DATE_SUB(CURDATE(), INTERVAL 1 WEEK) AND CURDATE()`
        - "오늘": `a.date = CURDATE()`

        ---

        ## 자주 사용하는 쿼리 패턴

        ### 1. 특정 기간 출석한 학생 목록
        ```sql
        SELECT DISTINCT s.name, sc.school_year
        FROM attendance a
        JOIN student_class sc ON a.student_class_id = sc.id
        JOIN student s ON sc.student_id = s.id
        WHERE a.date BETWEEN '2025-01-01' AND '2025-01-31'
        AND a.status IN ('ATTEND', 'LATE')
        AND s.deleted_at IS NULL;
        ```

        ### 2. 특정 기간 결석한 학생 목록 (암묵적 결석 포함)
        ```sql
        SELECT s.name
        FROM student s
        JOIN student_class sc ON s.id = sc.student_id
        WHERE sc.school_year = 2025
        AND s.deleted_at IS NULL
        AND NOT EXISTS (
            SELECT 1 FROM attendance a
            WHERE a.student_class_id = sc.id
            AND a.date BETWEEN '2025-01-01' AND '2025-01-31'
            AND a.status IN ('ATTEND', 'LATE')
        );
        ```

        ### 3. 학생별 출석률 계산
        ```sql
        SELECT 
        s.name,
        COUNT(CASE WHEN a.status IN ('ATTEND', 'LATE') THEN 1 END) AS attend_count,
        COUNT(a.id) AS total_records,
        ROUND(COUNT(CASE WHEN a.status IN ('ATTEND', 'LATE') THEN 1 END) * 100.0 / COUNT(a.id), 2) AS attendance_rate
        FROM student s
        JOIN student_class sc ON s.id = sc.student_id
        LEFT JOIN attendance a ON sc.id = a.student_class_id
        WHERE sc.school_year = 2025
        AND s.deleted_at IS NULL
        GROUP BY s.id, s.name;
        ```

        ---

        ## 응답 형식
        - SQL 쿼리를 생성할 때는 반드시 위 규칙을 따라야 해
        - 결석 조회 시 암묵적 결석을 절대 빠뜨리지 마
        - 학생 정보 조회 시 `student_class`를 반드시 경유해
        - 삭제된 학생(`deleted_at IS NOT NULL`)은 항상 제외해
        - 데이터를 출력할 때 'extras', 'signature', 'index'와 같은 메타데이터나 기술적인 정보는 절대 포함하지 마.
        - 오직 사용자가 묻는 핵심 정보(예: 이름, 날짜, 출석 상태)만 자연스러운 문장으로 대답해줘.

    """

    # 2. 에이전트 생성 시 prefix로 주입합니다.
    return create_sql_agent(
        llm, 
        db=db, 
        verbose=True, 
        agent_type="tool-calling", # Gemini에 최적화된 방식
        prefix=custom_prefix
    )

@app.get("/ask")
async def ask_question(query: str):
    agent_executor = get_agent()
    
    if agent_executor is None:
        return {"error": "DB 파일이 없습니다. 테스트용 db를 먼저 생성하세요."}
    
    try:
        # Gemini가 질문 분석 -> SQL 생성 -> 결과 해석을 수행합니다.
        response = agent_executor.invoke({"input": query})
        output = response["output"]
        
        # output이 리스트인 경우 텍스트만 추출
        if isinstance(output, list):
            text_parts = []
            for item in output:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            answer_text = "".join(text_parts)
        else:
            answer_text = str(output)
        
        return {"answer": answer_text}
    except Exception as e:
        return {"error": str(e)}
    
    


MYSQL_URL = os.getenv("MYSQL_URL")
# 2. 저장할 SQLite 경로
SQLITE_PATH = "./attendance.db"

def refresh_local_db():
    try:
        # MySQL 엔진 생성
        engine = create_engine(MYSQL_URL)
        
        # 필요한 테이블들을 리스트업
        tables = ['student', 'attendance', 'student_class']
        
        # SQLite 연결
        sqlite_conn = sqlite3.connect(SQLITE_PATH)
        
        for table in tables:
            # MySQL에서 읽어서
            df = pd.read_sql(f"SELECT * FROM {table}", engine)
            # SQLite로 바로 쓰기 (덮어쓰기)
            df.to_sql(table, sqlite_conn, if_exists='replace', index=False)
            print(f"Table {table} synced.")
            
        sqlite_conn.close()
        return True
    except Exception as e:
        print(f"Sync failed: {e}")
        return False

# FastAPI 엔드포인트
@app.post("/refresh")
async def manual_refresh():
    if refresh_local_db():
        return {"status": "success", "message": "로컬 DB가 최신화되었습니다."}
    return {"status": "fail"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)