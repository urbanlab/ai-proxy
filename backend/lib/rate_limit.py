import time


class rateLimit:
    def __init__(self):
        self.request_nb = 0
        self.request_limit = 0
        self.first_request_time = time.time()
        
    def is_allowed(self):
        current_time = time.time()
        seconds = current_time - self.first_request_time
        if seconds >= 60:
            self.first_request_time = time.time()
            self.request_nb = 0
        if self.request_nb > self.request_limit :
            return False

        print("INCRMENT REQUEST NB")
        print("req limit " + str(self.request_limit) + " | requestnb " + str(self.request_nb))
        print("SECONDES",seconds)

        self.request_nb += 1
        return True
