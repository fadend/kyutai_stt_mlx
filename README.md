This is based on edits to https://github.com/kyutai-labs/delayed-streams-modeling/blob/main/scripts/stt_from_file_mlx.py.

It's very much a work-in-progress. TODOs:
- Expose the model's support for timestamps and generate VTT.
- Indicate progress in processing audio. (`tail -f` on the output file works pretty well though.)
- If the token budget gets used up without fully processing the input, give an estimate of the extra tokens required. (remaining audio time * tokens used / audio time processed.)

It swaps out the dependency on [sphn (PyPI)](https://pypi.org/project/sphn/) ([Github](https://github.com/kyutai-labs/sphn))
for [soundfile](https://pypi.org/project/soundfile/) and [resamply](https://pypi.org/project/resampy/). Not sure if it should
have, but the sparse PyPI page for [sphn](https://pypi.org/project/sphn/) made me worry about including it as a dependency.

## Run

Example:

```
% uv run kyutai_stt_mlx_transcribe 2006_stat_dept_lecture.mp3 --output 2006_stat_dept_lecture.txt                           
```
