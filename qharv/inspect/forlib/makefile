NAME=pbcbox
export FC=gfortran
export CC=gcc
F2PY=f2py3

all:
	$(F2PY) -c -m $(NAME) $(NAME).f90
clean:
	rm -rf $(NAME)*.so __pycache__ *.pyc

# !$omp parallel do collapse(3) default(shared) private(jat,iat,j)
