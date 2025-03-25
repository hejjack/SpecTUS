#!/bin/bash

usage() {
	echo usage: $0 [-m model] [-e meta] [-i image] [-w workdir] smiles >&2
	exit 2
}

unset model meta image dir

while getopts m:e:i:w: opt; do case "$opt" in
	m) model="$OPTARG" ;;
	e) meta="$OPTARG" ;;
	i) image="$OPTARG" ;;	# must be absolute path
	w) dir="$OPTARG" ;;

	*) usage ;;
esac; done
shift $((OPTIND-1))

[ -z "$1" ] && usage

chunk="$1"
base=$(basename $chunk)

: ${SCRATCHDIR:=$TMPDIR}
: ${SCRATCHDIR:=/tmp}

cd "${dir:=$PBS_O_WORKDIR}" || exit 1
cp "$chunk" $SCRATCHDIR


if [ -z "$image" ]; then
	singularity pull $SCRATCHDIR/rassp.sif docker://ljocha/rassp:nvidia-2023-6
	image=$SCRATCHDIR/rassp.sif
fi

if [ -n "$model" ]; then
	cp "$model" $SCRATCHDIR/formulanet.model
	cp "$meta" $SCRATCHDIR/formulanet.meta
else
	md5sum -c - <<EOF
fac1e8124104363cfaec2dc5d0b93046  $SCRATCHDIR/formulanet.meta
3c19262f6fbcb5928afc9b7a35e0cd33  $SCRATCHDIR/formulanet.model
EOF

	if [ $? != 0 ]; then
		curl -k -o $SCRATCHDIR/formulanet.model 'https://people.cs.uchicago.edu/~ericj/rassp/formulanet_best_candidate_pcsim_pretrain.nist-fromscratch-3x9x128.35790555.00000740.model'
		curl -k -o $SCRATCHDIR/formulanet.meta 'https://people.cs.uchicago.edu/~ericj/rassp/formulanet_best_candidate_pcsim_pretrain.nist-fromscratch-3x9x128.35790555.meta'
	fi
fi

cp $(dirname $0)/rassp-predict.py $SCRATCHDIR

cd $SCRATCHDIR || exit 1

unset nv
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
	CUDA_VISIBLE_DEVICES=$(nvidia-smi -L | grep $CUDA_VISIBLE_DEVICES | sed 's/^GPU \([0-9]*\):.*/\1/')
	nv=--nv
fi

split -d -l 1000 ${base} ${base}_

# naming hack to keep _NN on two digits only
#for c in ${chunk}_900*; do
#	mv $c $(echo $c | sed 's/900\(.\)/9\1/')
#done

rm -f *.jsonl *-discard.smi

for c in ${base}_*; do
	singularity exec $nv -B $PWD:/work --pwd /work "$image" /opt/nvidia/nvidia_entrypoint.sh python3 rassp-predict.py -m formulanet.model -e formulanet.meta -w ${PBS_NCPUS:-1} -s "$c" -o "$c.jsonl" -d "$c-discard.smi"
done

cd $dir
cat $SCRATCHDIR/*.jsonl >"$chunk.jsonl"
cat $SCRATCHDIR/*-discard.smi >"$chunk-discard.smi"

# cleanup if needed
# cd $SCRATCHDIR && rm -rf *
