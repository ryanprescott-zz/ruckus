"""RUCKUS server main entry point."""

import uvicorn
from fastapi import FastAPI

app = FastAPI(title="RUCKUS Server", version="0.1.0")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "RUCKUS Server"}


def main():
    """Main entry point."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
