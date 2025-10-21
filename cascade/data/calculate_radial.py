from cascade.wavefunction import WaveFunction

def main():
    wf = WaveFunction(rmax=3000) 
    wf.save_radial_integrals()

if __name__ == '__main__':
    main()