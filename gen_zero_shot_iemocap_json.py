import random
import ollama
import re
import json
from huggingface_hub import login
#Login to your personal account of huggingface_hub
login("PERSONAL LOGIN ACCOUNT HERE")

descriptions = [['(Frustrated) The subject is at the Department of Motor Vehicles (DMV) and he/she is being sent back after standing in line for an hour for not having the right form of IDs', \
                 '(Angry) The subject works at DMV. He/she rejects the application.'], 
                ['(Sad) The subject, a new parent, was called to enroll the army in a foreign country. He/she has to separate from his/her spouse for more than 1 year.', \
                 '(Sad) The subject is his/her spouse and is extremely sad for the separation.'], 
                ['(Happy) The subject is telling his/her friend that he/she is getting married.', \
                 '(Happy) The subject is very happy and wants to know all the details of the proposal. He/she also wants to know the date of the wedding.'], 
                ['(Frustarted) The subject is unemployed and he/she has spent last 3 years looking for work in his/her area. He/she is losing hope.', \
                 '(Neutral) The subject is trying to encourage his/her friend.'], 
                ['(Angry) The subject is furious, because the airline lost his/her baggage and he/she will receive only $50 (for a new bag that cost over $150 and has lots of important things).', \
                 '(Neutral) The subject works for the airline. He/she tries to calm the customer.'], \
                ['(Sad) The subject is sad because a close friend died. He had cancer that was detected a year before his death.', \
                 '(Neutral) The subject is trying to support his friend in this difficult moment.'], 
                ['(Happy) The subject has been accepted at USC. He/she is telling this to his/her best friend.', \
                 '(Happy) The subject is very happy and wants to know the details (major, scholarship). He/she is also happy because he/she will stay in LA so they will be together.'], 
                ['(Neutral) He/She is trying to change the mood of the customer and solve the problem.', \
                 '(Angry) After 30 minutes talking with a machine, he/she is transferred to an operator. He/she expresses his/her frustration, but, finally, he/she changes his/her attitude.']]

def generate_prompt():
    subject_description_idx = random.sample([0, 1, 2, 3, 4, 5, 6, 7], 1)[0]
    subject_description = descriptions[subject_description_idx]
    instruction1 = 'You are a helpful assistant with human emotions and talking styles.\n'
    instruction2 = 'In the designed context of an interactive dyadic dialogue between a Male(M) and a Female(F).\n'
    instruction3 = 'Some scenarios are used for eliciting unscripted/unrehearsed interactions.'
    instruction4 = 'Following the scenario description below:\n'
    instruction5 = 'For subject 1: {}\n'.format(subject_description[0])
    instruction6 = 'For subject 2: {}\n'.format(subject_description[1])
    instruction7 = 'Generate a conversation between a Male(M) and a Female(F) under the scenario with CORRECT emotion labels. Label the emotion in [] at the end of the sentence.\n'
    instruction8 = 'Note that emotions can ONLY STRICTLY be chosen from Happy, Neutral, Sad, Frustrated or Angry.\nThe COMPLETE LINE BY LINE dialogue is:\n'
    prompt = ("").join([instruction1, instruction2, instruction3, instruction4, instruction5, \
                        instruction6, instruction7, instruction8])
    return subject_description_idx, prompt


SAVE_DIR = 'YOUR_OWN_PATH'

total_round = 550
cur_round = 0
effective_utt_num = 0
speaker = ['M:', 'F:']
while cur_round < total_round:
    subject_description_idx, input_prompt = generate_prompt()
    res = ollama.generate(model='llama3', prompt=input_prompt)
    content = ''
    for utt in res['response'].split('\n'):
        emotion = re.findall(r"\[([A-Za-z0-9_]+)\]", utt)
        if len(emotion) == 1 and emotion[0] in ['Angry', 'Neutral', 'Sad', 'Happy']:
            effective_utt_num += 1
            utt = ' '.join(utt.split())
            utt = utt.replace('[{}]'.format(emotion[0]), '')
            if ' '.join(utt.split())[:2] not in ['M:', 'F:']:
                content += '{} {} [{}]\n'.format(random.sample(speaker, 1)[0], ' '.join(utt.split()), emotion[0])
            else:
                content += '{} [{}]\n'.format(' '.join(utt.split()), emotion[0])
    if len(content):
        save_json = {'subject_description':subject_description_idx, 'text':content}
        with open('{}/para{:04d}.json'.format(SAVE_DIR, cur_round), 'w') as f:
            json.dump(save_json, f)
    cur_round += 1

print('total usable utterance number: {}'.format(effective_utt_num))