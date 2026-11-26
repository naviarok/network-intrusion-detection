"""Human-readable descriptions of the 42 UNSW-NB15 features, grouped as they appear on the page."""

CATEGORICAL = {
    "proto": "Transport protocol",
    "service": "Application service",
    "state": "Connection state",
}

# (group title, shown open?, [(column, description)]) - the order of the page
NUMERIC_GROUPS = [
    ("Volume and TTL", True, [
        ("dur", "Duration (s)"),
        ("spkts", "Packets, source → destination"),
        ("dpkts", "Packets, destination → source"),
        ("sbytes", "Bytes, source → destination"),
        ("dbytes", "Bytes, destination → source"),
        ("rate", "Packets per second"),
        ("sttl", "Source time-to-live"),
        ("dttl", "Destination time-to-live"),
        ("sload", "Source bits per second"),
        ("dload", "Destination bits per second"),
        ("sloss", "Source packets lost or resent"),
        ("dloss", "Destination packets lost or resent"),
    ]),
    ("Timing", False, [
        ("sinpkt", "Source inter-packet time (ms)"),
        ("dinpkt", "Destination inter-packet time (ms)"),
        ("sjit", "Source jitter (ms)"),
        ("djit", "Destination jitter (ms)"),
        ("tcprtt", "TCP round-trip time (s)"),
        ("synack", "SYN → SYN-ACK time (s)"),
        ("ackdat", "SYN-ACK → ACK time (s)"),
    ]),
    ("TCP and packet size", False, [
        ("swin", "Source TCP window"),
        ("dwin", "Destination TCP window"),
        ("stcpb", "Source TCP base sequence number"),
        ("dtcpb", "Destination TCP base sequence number"),
        ("smean", "Mean source packet size (bytes)"),
        ("dmean", "Mean destination packet size (bytes)"),
    ]),
    ("Application content", False, [
        ("trans_depth", "HTTP request/response depth"),
        ("response_body_len", "HTTP response body size (bytes)"),
        ("is_ftp_login", "FTP login with user and password"),
        ("ct_ftp_cmd", "FTP commands in the session"),
        ("ct_flw_http_mthd", "HTTP GET/POST requests in the session"),
    ]),
    ("Recent connections (last 100)", False, [
        ("ct_srv_src", "Same service and source"),
        ("ct_srv_dst", "Same service and destination"),
        ("ct_dst_ltm", "Same destination"),
        ("ct_src_ltm", "Same source"),
        ("ct_src_dport_ltm", "Same source and destination port"),
        ("ct_dst_sport_ltm", "Same destination and source port"),
        ("ct_dst_src_ltm", "Same source and destination"),
        ("ct_state_ttl", "State / TTL combination code"),
        ("is_sm_ips_ports", "Same IP and port on both sides (0/1)"),
    ]),
]
