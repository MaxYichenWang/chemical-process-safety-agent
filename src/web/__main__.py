"""Start the web server using the same configuration as its access policy."""

import uvicorn

from src.web.config import WEB_HOST, WEB_PORT


def main() -> None:
    # Query URLs can carry access tokens. Keep them out of access logs.
    uvicorn.run("src.web.app:app", host=WEB_HOST, port=WEB_PORT, access_log=False)


if __name__ == "__main__":
    main()
