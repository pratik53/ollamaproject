from threading import Thread, current_thread

def disp():
    print("Threading runnning",current_thread().name )


for i in range(10):
    t = Thread(target=disp)
    t.start()