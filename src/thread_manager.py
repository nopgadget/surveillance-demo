import threading

class ThreadManager:
    def __init__(self):
        self.stop_event = threading.Event()
        self.threads = []
    
    def add_thread(self, target, daemon=True):
        thread = threading.Thread(target=target)
        thread.daemon = daemon
        self.threads.append(thread)
        return thread
    
    def start_all(self):
        for thread in self.threads:
            thread.start()
    
    def stop_all(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=2) 