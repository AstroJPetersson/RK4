PY = /Users/jonathanpetersson/opt/anaconda3/bin/python

serial:
	$(PY) rk4_serial.py

profile:
	$(PY) -m cProfile -o o.profile rk4_serial.py
	$(PY) -m gprof2dot -f pstats o.profile | dot -Tpng -o profile.png

