import streamlit as st
from streamlit import session_state as ss
import db_utils
from pg_agents import PageAgents
from pg_tasks import PageTasks
from pg_crews import PageCrews
from pg_tools import PageTools
from pg_crew_run import PageCrewRun
from pg_export_crew import PageExportCrew
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from routes import routers
import logging

# Streamlit Configuration
st.set_page_config(page_title="CrewAI Studio", page_icon="img/favicon.ico", layout="wide")

logging.basicConfig(
    level=logging.DEBUG,  # Set log level to DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format the logs
    handlers=[
        logging.StreamHandler()  # Output logs to the terminal
    ]
)

# Create a logger instance
logger = logging.getLogger(__name__)

# FastAPI Setup
api_app = FastAPI()

from fastapi.middleware import Middleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

@api_app.exception_handler(RequestValidationError)
async def custom_validation_exception_handler(request: Request, exc: RequestValidationError):
    formatted_errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error.get("loc", []))
        formatted_errors.append({
            "field": field_path,
            "message": error.get("msg", "Invalid input"),
            "suggestion": ValidationErrorMiddleware.get_suggestion(error)
        })

    # Log the validation error
    logger.error(f"Validation Error: {formatted_errors}")

    return JSONResponse(
        status_code=422,
        content={
            "status": "failure",
            "error": "Validation Error",
            "message": "Some fields in your request are invalid. Please review the errors and try again.",
            "details": formatted_errors
        }
    )

class ValidationErrorMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            # Process the request as usual
            response = await call_next(request)
            return response
        except RequestValidationError as exc:
            # Build a user-friendly error response
            formatted_errors = []
            for error in exc.errors():
                # Extract field path
                field_path = " -> ".join(str(loc) for loc in error.get("loc", []))
                # Append detailed and actionable error
                formatted_errors.append({
                    "field": field_path,
                    "message": error.get("msg", "Invalid input"),
                    "suggestion": self.get_suggestion(error)  # Add actionable suggestions
                })

            # Log the validation error
            logger.error(f"Validation Error: {formatted_errors}")

            return JSONResponse(
                status_code=422,
                content={
                    "status": "failure",
                    "error": "Validation Error",
                    "message": "Some fields in your request are invalid. Please review the errors and try again.",
                    "details": formatted_errors
                }
            )
        except Exception as exc:
            # Handle unexpected errors
            logger.error(f"Unexpected Error: {str(exc)}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "failure",
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred. Please try again later.",
                    "details": str(exc)
                }
            )

    @staticmethod
    def get_suggestion(error):
        """Provide actionable suggestions based on the validation error."""
        if error["type"] == "missing":
            return "This field is required but was not provided."
        elif error["type"] == "string_type":
            return "This field should be a valid string."
        elif error["type"] == "value_error":
            return "This field contains an invalid value."
        else:
            return "Please ensure this field meets the required format or type."

# Add the custom middleware
api_app.add_middleware(ValidationErrorMiddleware)
# Add CORS Middleware
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include the router with all the routes
# api_app.include_router(router)
# Add all routers to the app
for router in routers:
    api_app.include_router(router)
# Run FastAPI in a separate thread
def run_fastapi():
    uvicorn.run(api_app, host="0.0.0.0", port=8000, log_level="debug")

def pages():
    return {
        'Crews': PageCrews(),
        'Tools': PageTools(),
        'Agents': PageAgents(),
        'Tasks': PageTasks(),
        'Kickoff!': PageCrewRun(),
        'Import/export': PageExportCrew()
    }

def load_data():
    ss.agents = db_utils.load_agents()
    ss.tasks = db_utils.load_tasks()
    ss.crews = db_utils.load_crews()
    ss.tools = db_utils.load_tools()
    ss.enabled_tools = db_utils.load_tools_state()


def draw_sidebar():
    with st.sidebar:
        st.image("img/crewai_logo.png")

        if 'page' not in ss:
            ss.page = 'Crews'
        
        selected_page = st.radio('Page', list(pages().keys()), index=list(pages().keys()).index(ss.page),label_visibility="collapsed")
        if selected_page != ss.page:
            ss.page = selected_page
            st.rerun()

def main():
    load_dotenv()
    if (str(os.getenv('AGENTOPS_ENABLED')).lower() in ['true', '1']) and not ss.get('agentops_failed', False):
        try:
            import agentops
            agentops.init(api_key=os.getenv('AGENTOPS_API_KEY'),auto_start_session=False)    
        except ModuleNotFoundError as e:
            ss.agentops_failed = True
            print(f"Error initializing AgentOps: {str(e)}")            
        
    db_utils.initialize_db()
    load_data()
    draw_sidebar()
    PageCrewRun.maintain_session_state() #this will persist the session state for the crew run page so crew run can be run in a separate thread
    pages()[ss.page].draw()
    
if __name__ == '__main__':
    main()
