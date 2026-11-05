cd /media/shawn/C/papers/VLM/judgebench
P=/home/shawn/anaconda3/envs/lis-omni/bin/python; S=/media/shawn/C/papers/EAGLE-VLM/scripts/probe_ties.py; R=/media/shawn/C/papers/VLM/judgebench_local/runs
for j in qwen3vl8b internvl35_8b; do
  $P $S --runs $R/H1_synth_hidden/raw --judge $j --ds synth_tie --out paper_tables/probe_${j}_H1g.md > paper_tables/probe_${j}_H1g.log 2>&1
  $P $S --runs $R/H3_synth_blind_hidden/raw --judge $j --ds synth_tie --blind --out paper_tables/probe_${j}_H3blind.md > paper_tables/probe_${j}_H3blind.log 2>&1
done
for j in llavaov7b qwen25vl7b r1reward; do
  $P $S --runs $R/H1_synth_hidden/raw --judge $j --ds synth_tie --out paper_tables/probe_${j}_H1g.md > paper_tables/probe_${j}_H1g.log 2>&1
done
echo PROBE_SYNTH_DONE > paper_tables/probe_synth.done
