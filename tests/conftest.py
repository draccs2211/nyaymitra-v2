"""
Test-wide setup. Runs before any backend import.
Disable LangSmith tracing: tests must not upload (mocked) runs to LangSmith,
and the uploads stall on network timeouts. load_dotenv() won't override these.
"""
import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
