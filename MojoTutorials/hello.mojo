import math

let PI = 3.141592653589793115997963468544185161590576171875

fn main():
    print('Hello')
    print(convolve(4, 7))
    let c = Circle(3.5)
    print('Area of Circle (with radius ' + str(c.radius) + ') = ' + c.area())
    print('Circumference of Circle (with radius ' + str(c.radius) + ') = ' + c.circumference())

fn convolve(x: Float32, y: Float32) -> Float32:
    return x * y


struct Circle:
    var radius: Float32

    fn __init__(inout self, radius: Float32):
        self.radius = radius

    fn area(self) -> Float32:
        return PI *  self.radius ** 2

    fn circumference(self) -> Float32:
        return 2 * PI * self.radius


