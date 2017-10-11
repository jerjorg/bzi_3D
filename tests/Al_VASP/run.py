"""I believe Matt is running VASP and then removing the WAVECAR and CHGCAR.
"""
#!/apps/python/3.3.2/bin/python
import subprocess, os

# subprocess.call('vasp53s')
subprocess.call('vasp.x')
os.remove('WAVECAR')
os.remove('CHGCAR')
