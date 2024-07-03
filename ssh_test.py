import paramiko
import paramiko.util

# Enable debug level logs
paramiko.util.log_to_file('./log', level='DEBUG')

private_key_file = "/Users/lukas/.ssh/id_ed25519"
private_key_file = "/Users/lukas/.ssh/id_ed_bwunicluster"
password = "password"
host = "uc2.scc.kit.edu"
user = "fq0795"
port = 23



if __name__ == "__main__":
    k = paramiko.Ed25519Key.from_private_key_file(private_key_file, password)
    
    ssh = paramiko.SSHClient()

    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=host, username=user, pkey=k, disabled_algorithms={
        "pubkeys": ["rsa-sha2-256", "rsa-sha2-512"]
    })

    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("ls -a")
    print(ssh_stdin)
    print(ssh_stdout)
    print(ssh_stderr)