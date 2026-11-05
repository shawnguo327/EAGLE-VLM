cd /media/shawn/C/papers/VLM/judgebench
L=/media/shawn/C/papers/VLM/judgebench_local/runs/scheduler.log
until grep -q "end H2_real_hidden/r1reward" $L; do sleep 120; done
P=/home/shawn/anaconda3/envs/lis-omni/bin/python; S=/media/shawn/C/papers/EAGLE-VLM/scripts/probe_ties.py; R=/media/shawn/C/papers/VLM/judgebench_local/runs/H2_real_hidden/raw
for j in internvl35_8b qwen25vl7b llavaov7b r1reward; do
  $P $S --runs $R --judge $j --ds genai_arena,genai_arena:edition,genai_arena:generation,mllm_judge --out paper_tables/probe_${j}_H2.md > paper_tables/probe_${j}_H2.log 2>&1
done
echo done > paper_tables/probe_real.done
