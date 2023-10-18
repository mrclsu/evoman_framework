import subprocess

def run_with_enemy(enemy):
    processes = []
    for i in range(0, 10):
        command = f"python3 deap_generalist.py --train --instance {i} --enemy {enemy}"
        process = subprocess.Popen(command.split())
        processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        process.wait()


def main():
    for enemy in range(0, 2):
        run_with_enemy(enemy)

if __name__ == '__main__':
    main()
