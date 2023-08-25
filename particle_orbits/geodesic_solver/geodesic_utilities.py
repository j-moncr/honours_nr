import numpy

class dual:
    def __init__(self, first, second):
        self.f = first
        self.s = second

    def __mul__(self,other):
        if isinstance(other,dual):
            return dual(self.f*other.f, self.s*other.f+self.f*other.s)
        else:
            return dual(self.f*other, self.s*other)

    def __rmul__(self,other):
        if isinstance(other,dual):
            return dual(self.f*other.f, self.s*other.f+self.f*other.s)
        else:
            return dual(self.f*other, self.s*other)

    def __add__(self,other):
        if isinstance(other,dual):
            return dual(self.f+other.f, self.s+other.s)
        else:
            return dual(self.f+other,self.s)

    def __radd__(self,other):
        if isinstance(other,dual):
            return dual(self.f+other.f, self.s+other.s)
        else:
            return dual(self.f+other,self.s)

    def __sub__(self,other):
        if isinstance(other,dual):
            return dual(self.f-other.f, self.s-other.s)
        else:
            return dual(self.f-other,self.s)

    def __rsub__(self, other):
        return dual(other, 0) - self

    def __truediv__(self,other):
        ''' when the first component of the divisor is not 0 '''
        if isinstance(other,dual):
            return dual(self.f/other.f, (self.s*other.f-self.f*other.s)/(other.f**2.))
        else:
            return dual(self.f/other, self.s/other)

    def __rtruediv__(self, other):
        return dual(other, 0).__truediv__(self)

    def __neg__(self):
        return dual(-self.f, -self.s)

    def __pow__(self, power):
        return dual(self.f**power,self.s * power * self.f**(power - 1))
    
    def sqrt(self):
        # return dual(self ** 0.5, self.s * 0.5 * self.f**(-0.5))
        return pow(self, 0.5)

    def sin(self):
        return dual(numpy.sin(self.f),self.s*numpy.cos(self.f))

    def cos(self):
        return dual(numpy.cos(self.f),-self.s*numpy.sin(self.f))

    def tan(self):
        return sin(self)/cos(self)

    def log(self):
        return dual(numpy.log(self.f),self.s/self.f)

    def exp(self):
        return dual(numpy.exp(self.f),self.s*numpy.exp(self.f))

def dif(func,x):
    funcdual = func(dual(x,1.))
    if isinstance(funcdual,dual):
        return func(dual(x,1.)).s
    else:
        ''' this is for when the function is a constant, e.g. gtt:=0 '''
        return 0