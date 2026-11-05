"""Minimal, env-gated patch of VLMEvalKit's official InternVL Best-of-N path: when BON_DUMP is set, also record all N
sampled candidates and VisualPRM's scores (same select_best_response call with return_scores=True, so the selected
answer is unchanged).  Without BON_DUMP the code path is the original one."""
import sys
p = sys.argv[1]; s = open(p).read()
old = """        if self.best_of_n > 1:
            response_list = self.reward_model.select_best_response(
                tokenizer=self.reward_tokenizer,
                question=prompt,
                response_list=response_list,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
            )
        response = response_list[0]
"""
new = """        if self.best_of_n > 1 and os.environ.get('BON_DUMP'):          # EAGLE-VLM patch: dump candidates + PRM scores
            import json as _json
            _cands = list(response_list)
            _scored = self.reward_model.select_best_response(
                tokenizer=self.reward_tokenizer,
                question=prompt,
                response_list=response_list,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                return_scores=True,
            )
            response_list = [r for r, _ in _scored]
            with open(os.environ['BON_DUMP'], 'a') as _f:
                _f.write(_json.dumps({'dataset': dataset, 'prompt': prompt,
                                      'images': [x['value'] for x in message if x['type'] == 'image'],
                                      'candidates': _cands, 'prm_sorted': [[r, float(sc)] for r, sc in _scored]},
                                     ensure_ascii=False) + '\\n')
        elif self.best_of_n > 1:
            response_list = self.reward_model.select_best_response(
                tokenizer=self.reward_tokenizer,
                question=prompt,
                response_list=response_list,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
            )
        response = response_list[0]
"""
assert s.count(old) == 1, "patch target not found exactly once"
open(p, "w").write(s.replace(old, new)); print("patched", p)
