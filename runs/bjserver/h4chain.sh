L=/media/shawn/C/papers/VLM/judgebench_local/runs/scheduler.log
P=/home/shawn/anaconda3/envs/lis-omni/bin/python; C=/media/shawn/C/papers/VLM/judgebench_local/config.yaml
until grep -q "end L4_scale_3b/qwen25vl3b" $L; do sleep 60; done
cd /media/shawn/C/papers/EAGLE-VLM/pack && $P ctl.py --config $C stop; sleep 20
$P ctl.py --config $C start H4_ft_hidden L4_scale_3b < /dev/null > /dev/null 2>&1
until grep -q "end H4_ft_hidden/qwen35_9b" $L; do sleep 120; done
cd /media/shawn/C/papers/VLM/judgebench
for j in llavacritic7b mmrlhfrm7b qwen35_9b; do
  $P /media/shawn/C/papers/EAGLE-VLM/scripts/probe_ties.py --runs /media/shawn/C/papers/VLM/judgebench_local/runs/H4_ft_hidden/raw --judge $j --ds synth_tie --out paper_tables/probe_${j}_H4.md > paper_tables/probe_${j}_H4.log 2>&1
done
$P /media/shawn/C/papers/EAGLE-VLM/scripts/levels.py --out paper_tables/levels.md > /dev/null 2>&1
echo done > paper_tables/h4.done
