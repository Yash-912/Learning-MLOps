import pytest

def square(n):
    return n*n
def cube(n):
    return n**3
def fifth(n):
    return n**5

def test_square():
    assert square(2)==4
    assert square(3)==9
    
def test_cube():
    assert cube(2)==4
    assert cube(3)==9

def test_fifth():
    assert fifth(2)==32
    assert fifth(3)==243
    
def test_invalid_input():
    with pytest.raises(TypeError):
        square("string")
    