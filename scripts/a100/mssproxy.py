# ssh ProxyCommand: plain TCP relay whose socket uses a small MSS (works around a path-MTU black hole); client side only
import os, select, socket, sys
host, port, mss = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 1200
s = socket.socket(); s.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, mss); s.connect((host, port))
i, o = sys.stdin.fileno(), sys.stdout.fileno()
while True:
    r, _, _ = select.select([s, i], [], [])
    if s in r:
        b = s.recv(65536)
        if not b: break
        os.write(o, b)
    if i in r:
        b = os.read(i, 65536)
        if not b: s.shutdown(socket.SHUT_WR); continue
        s.sendall(b)
