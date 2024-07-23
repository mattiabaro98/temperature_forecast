import time
import threading

a = 0

def update_model():
    global a
    print("doing something")
    a += 1

def run_periodic_task():
    while True:
        update_model()
        time.sleep(12)

def main():
    global a
    while True:
        print("Hi")
        print(a)
        time.sleep(1)

if __name__ == "__main__":
    task_thread = threading.Thread(target=run_periodic_task)
    task_thread.start()
    main()
