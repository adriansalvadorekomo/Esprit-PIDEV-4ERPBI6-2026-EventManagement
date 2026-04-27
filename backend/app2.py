from eventzilla_api.app import create_app

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app2:app", host="0.0.0.0", port=8000, reload=True)
