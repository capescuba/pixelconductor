"""
PixelConductor launcher — run with pythonw.exe so no console window appears.

  pythonw launch.pyw          start the server (if not already running) and open the browser
  pythonw launch.pyw --stop   stop the background server

Server output goes to logs/server.log.
"""
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = 7842
URL  = f"http://localhost:{PORT}"
LOG  = HERE / "logs" / "server.log"


def server_up():
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=0.5):
            return True
    except OSError:
        return False


def message(text):
    import ctypes
    ctypes.windll.user32.MessageBoxW(None, text, "PixelConductor", 0x10)


def stop():
    # Kill whatever is listening on the port (the hidden app.py process).
    out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True,
                         creationflags=subprocess.CREATE_NO_WINDOW).stdout
    pids = {line.split()[-1] for line in out.splitlines()
            if f":{PORT} " in line and "LISTENING" in line}
    for pid in pids:
        subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)


def start():
    if not server_up():
        LOG.parent.mkdir(exist_ok=True)
        log = open(LOG, "a", encoding="utf-8")
        log.write(f"\n===== launched {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        log.flush()
        # pythonw.exe next to the interpreter running this launcher
        exe = Path(sys.executable).with_name("pythonw.exe")
        subprocess.Popen(
            [str(exe if exe.exists() else sys.executable), str(HERE / "app.py")],
            cwd=HERE, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        deadline = time.time() + 60
        while not server_up():
            if time.time() > deadline:
                message(f"PixelConductor server didn't start within 60s.\n\nSee {LOG}")
                return
            time.sleep(0.25)
    webbrowser.open(URL)


if __name__ == "__main__":
    stop() if "--stop" in sys.argv else start()
