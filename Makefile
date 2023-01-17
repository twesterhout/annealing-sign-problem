
SPIN_ED = /vol/tcm01/westerhout_tom/spin-ed/SpinED-4c3305a
PYTHON = python3
SEED = 435834


all:

.PHONY: small
# small: experiments/j1j2_square_4x4.csv
	
small: experiments/heisenberg_kagome_16.csv \
	experiments/heisenberg_kagome_18.csv \
	experiments/j1j2_square_4x4.csv \
	experiments/sk_16_1.csv \
	experiments/sk_16_2.csv \
	experiments/sk_16_3.csv

.PHONY: ed
ed: physical_systems/heisenberg_kagome_16.h5 \
    physical_systems/heisenberg_kagome_18.h5 \
    physical_systems/j1j2_square_4x4.h5 \
    physical_systems/sk_16_1.h5 \
    physical_systems/sk_16_2.h5 \
    physical_systems/sk_16_3.h5

physical_systems/%.h5: physical_systems/%.yaml
	OMP_NUM_THREADS=`nproc` $(SPIN_ED) $< && mv $(@F) $(@D)

experiments/%.csv: ed
	$(PYTHON) experiments/full_hilbert_space.py \
		--hdf5 physical_systems/$(*F).h5 \
		--yaml physical_systems/$(*F).yaml \
		--seed $(SEED) \
		--output $@.wip \
		--number-sweeps 100,200,400,800,1600,3200,6400,12800,25600,51200,102400,204800 \
		--repetitions 1024 && \
	mv $@.wip $@

.PHONY: pyrochlore_32
pyrochlore_32:
	$(PYTHON) experiments/sampled_connected_components.py \
		--system pyrochlore