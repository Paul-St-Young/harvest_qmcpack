NAME=stats
export FC=gfortran
export CC=gcc

all:
	f2py -c -m $(NAME) $(NAME).f90
clean:
	rm -rf $(NAME)*.so __pycache__ *.pyc
