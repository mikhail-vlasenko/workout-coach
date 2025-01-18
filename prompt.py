from stages import STAGES

STAGE_MISTAKES = {}

for stage in STAGES:
    with open(f'data/{stage}_mistakes.txt', 'r') as f:
        STAGE_MISTAKES[stage] = ''.join(f.readlines())


def get_prompt(stage):
    prompt = f'The image presents a person doing a deadlift. The person is in the {stage} stage of the exercise.'
    prompt += '\nHere is a numbered list of the common mistakes that can be made during this stage:\n'
    prompt += STAGE_MISTAKES[stage]
    prompt += ('\nIf the person is making any of these mistakes, provide the corresponding number(s) in the response. '
               'Separate multiple numbers with commas. Do not include any other text in the response.')
    # prompt = "describe the pose of the person in the image"
    return prompt
