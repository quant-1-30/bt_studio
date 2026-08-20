#!/usr/bin/env python3
"""Qt GUI entry — the desktop client has been extracted to bt_clients repo."""

import os
import sys


def main():
    print("=" * 64)
    print("The Qt desktop client is no longer part of bt_studio.")
    print()
    print("It was extracted to the separate client repository:")
    print("    ~/startup/bt_clients/  (bt_clients)")
    print()
    print("Server-side, start the results gateway instead:")
    print("    uvicorn bt_studio.plugins.api_server.main:app "
          "--host 0.0.0.0 --port 8000")
    print("The Qt app connects to it via HTTP + WebSocket.")
    print("=" * 64)
    sys.exit(1)


if __name__ == "__main__":
    main()