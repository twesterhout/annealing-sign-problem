
SPIN_ED = /vol/tcm01/westerhout_tom/spin-ed/SpinED-4c3305a
PYTHON = python3
INPUT_DATA_URL = https://surfdrive.surf.nl/files/index.php/s/Ec5CILNO5tbXlVk/download
JOBID =
NOISE = 0

ifneq ($(JOBID),)
  SEED = $(JOBID)
else
  SEED = 435834
endif

ifeq ($(NOISE),0)
  EXTRA_PREFIX = no_noise
else
  EXTRA_PREFIX = noise_$(NOISE)
endif


all:

.PHONY: small
# small: experiments/j1j2_square_4x4.csv
	
small: experiments/heisenberg_kagome_16.csv \
	experiments/heisenberg_kagome_18.csv \
	experiments/j1j2_square_4x4.csv \
	experiments/sk_16_1.csv \
	experiments/sk_16_2.csv \
	experiments/sk_16_3.csv

experiments/%.csv: physical_systems/data-small/%.h5
	$(PYTHON) experiments/full_hilbert_space.py \
		--hdf5 physical_systems/data-small/$(*F).h5 \
		--yaml physical_systems/$(*F).yaml \
		--seed $(SEED) \
		--output $@.wip \
		--number-sweeps 100,200,400,800,1600,3200,6400,12800,25600,51200,102400,204800 \
		--repetitions 1024 && \
	mv $@.wip $@

.PHONY: pyrochlore_32
pyrochlore_32:
	@mkdir -p experiments/pyrochlore/$(EXTRA_PREFIX)
	$(PYTHON) experiments/sampled_connected_components.py \
		--hdf5 physical_systems/data-large/heisenberg_pyrochlore_2x2x2.h5 \
		--yaml physical_systems/heisenberg_pyrochlore_2x2x2.yaml \
		--seed $(SEED) \
		--output experiments/pyrochlore/$(EXTRA_PREFIX)/pyrochlore_32.csv$(JOBID) \
		--order 2 \
		--noise $(NOISE) \
		--no-annealing \
		--global-cutoff 1e-5 \
		--number-samples 10000

.PHONY: kagome_36
kagome_36:
	@mkdir -p experiments/kagome/$(EXTRA_PREFIX)
	$(PYTHON) experiments/sampled_connected_components.py \
		--hdf5 physical_systems/data-large/heisenberg_kagome_36.h5 \
		--yaml physical_systems/heisenberg_kagome_36.yaml \
		--seed $(SEED) \
		--output experiments/kagome/$(EXTRA_PREFIX)/kagome_36.csv$(JOBID) \
		--order 3 \
		--noise $(NOISE) \
		--no-annealing \
		--global-cutoff 2e-6 \
		--number-samples 10000

.PHONY: sk_32_1
sk_32_1:
	$(PYTHON) experiments/sampled_connected_components.py \
		--hdf5 physical_systems/data-large/sk_32_1.h5 \
		--yaml physical_systems/sk_32_1.yaml \
		--seed $(SEED) \
		--output experiments/sk_32_1.csv.wip$(JOBID) \
		--order 3 \
		--noise $(NOISE) \
		--no-annealing \
		--global-cutoff 1e-5 \
		--number-samples 10

physical_systems/data-small:
	mkdir -p $(@D) && \
	cd $(@D) && \
	wget --no-verbose -O tmp.zip $(INPUT_DATA_URL)?path=/physical_systems/data-small && \
	unzip tmp.zip && rm tmp.zip

physical_systems/data-large:
	mkdir -p $(@D) && \
	cd $(@D) && \
	wget --no-verbose -O tmp.zip $(INPUT_DATA_URL)?path=/physical_systems/data-large && \
	unzip tmp.zip && rm tmp.zip

# Initiall the hdf5 files were generated using the following rules:
#
# .PHONY: ed
# ed: physical_systems/heisenberg_kagome_16.h5 \
#     physical_systems/heisenberg_kagome_18.h5 \
#     physical_systems/j1j2_square_4x4.h5 \
#     physical_systems/sk_16_1.h5 \
#     physical_systems/sk_16_2.h5 \
#     physical_systems/sk_16_3.h5
# 
# physical_systems/%.h5: physical_systems/%.yaml
# 	OMP_NUM_THREADS=`nproc` $(SPIN_ED) $< && mv $(@F) $(@D)
