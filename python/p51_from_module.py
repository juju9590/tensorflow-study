from machine.car import drive
from machine.tv import watch

drive() #운전하다
watch() #시청하다

print("=============================")
from machine import car, tv

car.drive()
tv.watch()

# machine 운전하다
# machine 시청하다
# =============================
# machine 운전하다
# machine 시청하다

print("======== test 폴더 ==========")


from machine.test.car import drive
from machine.test.tv import watch

drive() #운전하다
watch() #시청하다

print("=============================")
from machine.test import car, tv

car.drive()
tv.watch()

# ============= test 폴더 ================
# machine.test 운전하다
# machine.test 시청하다
# =============================
# machine.test 운전하다
# machine.test 시청하다

