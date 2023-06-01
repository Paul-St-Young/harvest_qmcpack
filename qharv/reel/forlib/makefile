NAME=stats
export FC=gfortran
export CC=gcc

all:
	f2py3 -c -m $(NAME) $(NAME).f90
clean:
	rm -rf $(NAME)*.so __pycache__ *.pyc
