PY = /Users/jonathanpetersson/opt/anaconda3/bin/python
MPI = /Users/jonathanpetersson/opt/anaconda3/bin/mpiexec

serial:
	$(PY) rk4_serial.py

profile:
	$(PY) -m cProfile -o profile.out rk4_serial.py
	$(PY) -m gprof2dot -f pstats profile.out | dot -Tpng -o profile.png

mpi:
	$(MPI) -n 1 $(PY) rk4_mpi.py


