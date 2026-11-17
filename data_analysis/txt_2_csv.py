#python3 -m venv venv
#source venv/bin/activate

import pandas as pd

input_file = "smotenc_generated_samples_labels_concatenated.txt"
output_file = "smotenc_generated_samples_labels_concatenated.csv"

columns = [
    "dur", "proto", "service", "state", "spkts", "dpkts",
    "sbytes", "dbytes", "rate", "sttl", "dttl", "sload",
    "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit",
    "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
    "synack", "ackdat", "smean", "dmean", "trans_depth",
    "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports", "attack_cat"
]

df = pd.read_csv(
    input_file,
    sep=r"\s+",
    header=None,
    skiprows=1,
    usecols=range(1, len(columns) + 1),
    engine="python"
)

df.columns = columns

df.to_csv(output_file, index=False)

print(f"Conversão concluída! Arquivo gerado: {output_file}")
print(f"Total de registros: {len(df)}")
print(f"Total de colunas: {len(df.columns)}")
