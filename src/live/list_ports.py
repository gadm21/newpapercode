import serial.tools.list_ports

ports = serial.tools.list_ports.comports()
for p in ports:
    vid = f"{p.vid:04X}" if p.vid else "None"
    pid = f"{p.pid:04X}" if p.pid else "None"
    print(f"{p.device}: {p.description} (VID:PID={vid}:{pid})")
