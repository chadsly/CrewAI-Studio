from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import threading
import queue
import time
from pg_crew_run import PageCrewRun
import db_utils
from my_crew import MyCrew
from routes.db_routes import router as db_router

# Create a FastAPI router
router = APIRouter()

# Include database routes
router.include_router(db_router, prefix="/db", tags=["Database"])

class CrewRunPayload(BaseModel):
    crew_id: str
    inputs: dict = {}  # Add inputs if necessary

# Initialize DB on FastAPI Startup
@router.on_event("startup")
async def startup_event():
    print("Initializing database for FastAPI...")
    db_utils.initialize_db()

# Define Routes
@router.get("/api")
def hello():
    return {"Hello World"}

@router.get("/api/crews")
def get_crews():
    crews = db_utils.load_crews()
    print(crews)
    return {"crews": crews}

@router.post("/api/crews/run")
def run_crew(payload: CrewRunPayload):
    try:
        print("Starting run for crew: {payload.crew_id}...")
        crews = db_utils.load_crews()  # Load crews from a shared datastore
        selected_crew = next((crew for crew in crews if crew.id == payload.crew_id), None)
        if not selected_crew:
            raise HTTPException(status_code=404, detail=f"Crew with ID {payload.crew_id} not found.")
        print("Running crew: {selected_crew}")

        crew_runner = PageCrewRun()
        message_queue = queue.Queue()
        print("Message Queue created.")
        # Start the crew runner thread
        print("Starting crew runner thread...")
        crew_thread = threading.Thread(
            target=crew_runner.run_crew,
            kwargs={
                "crewai_crew": selected_crew.get_crewai_crew(),
                "inputs": {},
                "message_queue": message_queue
            }
        )
        print("Thread started.")
        crew_thread.start()

        # Wait briefly for the thread to start and catch any errors
        time.sleep(0.5)  # Allow thread to populate the queue
        try:
            # Check for any immediate errors in the message queue
            message = message_queue.get_nowait()
            if "error" in message:
                raise HTTPException(status_code=500, detail=message["error"])
        except queue.Empty:
            pass  # No errors yet, thread is running successfully

        return {"status": "running", "crew_id": payload.crew_id}

    except Exception as e:
        # Catch any exceptions and return them as the response
        raise HTTPException(status_code=500, detail=str(e))

