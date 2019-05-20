import numpy as np


from clause import Clause, Question
from oracle import Oracle
from dynamic_actions import *
from collections import defaultdict
import random

def sample_question(oracle_start_state, oracle, agent1, agent2, obj, question):
    idx_dummy = [0]
    if question == 'memory':
        action, trace = MemoryAction(oracle_start_state, obj), 'memory'
    elif question == 'reality':
        action, trace = RealityAction(oracle, obj), 'reality'
    elif question == 'belief':
        action = BeliefSearchAction(oracle, agent1.name, agent2.name, obj)
        trace = f'second_order_{agent1.order}_{"" if action.tom else "no_"}tom'
    elif question == 'search':
        action = SearchedAction(oracle, agent1.name, obj)
        trace = f'first_order_{agent1.order}_{"" if action.tom else "no_"}tom'
    return Question(idx_dummy, action), trace
# -------------------------------- Chapters ---------------------------------- #

class Agent:
    def __init__(self, id : int, name : str, order : int):
        self.id = id + 1  # these are 1 indexed for some reason...
        self.name = name
        self.order = order

def ids(agents):
    return [agent.id for agent in agents]

def names(agents):
    return [agent.name for agent in agents]

def enter(oracle, agent, observers, location):
    if oracle.get_location(agent.name) == location:  # already in location
        return LocationAction(oracle, (agent.name, location))
    else:  # somewhere else, move this person into location
        return EnterAction(oracle, (agent.name, location), names(observers))

def write_false_belief_chapter(
    start_state, oracle, location, agent_ids, all_agents,
    alternative_loc, question
):
    """
    Creates list of clauses that constitute
    a true belief task.

    agent_ids: list that gives indices of agents
      in container. should be length 2.
    all_agents: list of all agents
    container: container to which the object is
      moved

    Warning: clauses will advance that state
    of the simulation, should clauses should
    be appended in order.
    """
    a1, a2, a3 = tuple(Agent(id, all_agents[id], i) for i, id in enumerate(agent_ids))

    # pick random object at location
    obj = np.random.choice(oracle.get_objects_at_location(location))
    container_1 = oracle.get_object_container(obj)

    # pick random container in locations
    container_candidates = oracle.get_containers(location)[:]
    container_candidates.remove(container_1)
    container_2 = np.random.choice(container_candidates)
    trace = []
    chapter = []

    # randomize the order in which agents enter the room
    agents = [a1, a2]
    enter_observers = []
    random.shuffle(agents)
    for agent in agents:
        action = enter(oracle, agent, enter_observers, location)
        chapter.append(Clause(ids(enter_observers) + [agent.id], action))
        enter_observers.append(agent)
        trace.append(f'enter_agent_{agent.order}')

    # announce location of object
    chapter.append(Clause(ids(agents), ObjectLocAction(oracle, obj, names(agents))))

    act_types = ['move'] + ['loc_change'] * random.randint(1, 2)
    random.shuffle(act_types)
    move_observers = [a2]
    for act_type in act_types:
        if act_type == 'move':
            # move the object to container_2
            act = MoveAction(oracle, (a1.name, obj, container_2), names(move_observers))
            chapter.append(Clause(ids(move_observers), act))
            trace.append(f'agent_{a1.order}_moves_obj')
        elif oracle.get_location(a2.name) == location:
            # a2 is in location, exit...
            chapter.append(Clause([a1.id], ExitedAction(oracle, (a2.name))))
            move_observers = []
            trace.append(f'agent_{a2.order}_exits')
        else:
            enter_loc = location if random.randint(0, 1) == 0 else alternative_loc
            # a2 already existed, re-enter same room, or a different one
            chapter.append(Clause([a1.id], EnterAction(oracle, (a2.name, enter_loc), [a1.name])))
            move_observers = [a2]
            trace.append(f'agent_{a2.order}_reenters_' + ('alt_loc' if enter_loc != location else 'loc'))


    # generate indices for which person 3 should enter/exit
    indices = np.random.choice(np.arange(len(chapter)+1), replace=False, size=random.randint(0, 2))
    indices.sort()
    for idx, action in zip(indices, ['enter', 'exit']):
        if action == 'exit':
            clause = Clause(ids(move_observers), ExitedAction(oracle, (a3.name)))
            enter_observers.pop()  # remove person 3 from observers
            trace.insert(idx, f'agent_{a3.order}_exits')
            chapter.insert(idx, clause)
        else:
            enter_loc = location if random.randint(0, 1) == 0 else alternative_loc
            clause = Clause(ids(move_observers), EnterAction(oracle, (a3.name, enter_loc), names(enter_observers)))
            enter_observers.append(a3)
            trace.insert(idx, f'agent_{a3.order}_enters')
            chapter.insert(idx, clause)

    # Add noise:
    indices = np.random.choice(np.arange(len(chapter)+1), replace=False, size=random.randint(0, 2))
    for idx in indices:
        person = np.random.choice([a1, a2, a3], 1)[0].name
        things = list(oracle.locations.containers.keys())
        things += list(oracle.locations.obj_containers.keys())
        thing = np.random.choice(things, 1)[0]
        chapter.insert(idx, Clause([], ActualNoise(oracle, person, thing)))

    if question == 'all':
        stories, traces = [], []
        for q in ['memory', 'search', 'belief', 'reality']:
            qtext, qtrace = sample_question(start_state, oracle, a1, a2, obj, q)
            stories.append(chapter + [qtext])
            traces.append(trace + [qtrace])
        for q in ['search', 'belief']:
            qtext, qtrace = sample_question(start_state, oracle, a2, a1, obj, q)
            stories.append(chapter + [qtext])
            traces.append(trace + [qtrace])
        return stories, traces
    else:
        agents = [a1, a2]
        random.shuffle(agents)
        qtext, qtrace = sample_question(start_state, oracle, agents[0],
            agents[1], obj, question)
        chapter.append(qtext)
        trace.append(qtrace)
        return [chapter], [trace]


#######################################
############### Tasks #################
#######################################

class Specify_Tasks:
    def __init__(self,
                 num_questions=5,
                 exit_prob=1.,
                 informant_prob=1.,
                 search_prob=1.):

        self.num_questions = num_questions
        self.search_prob = search_prob
        self.exit_inform_probs = [1 - exit_prob,
                                  exit_prob * (1 - informant_prob),
                                  exit_prob * informant_prob]
        assert sum(self.exit_inform_probs) == 1

    def generate_story(
        self, world, task, question, num_agents=6,
        num_locations=3, statement_noise=0
    ):
        """
        Allows user to specify chapter and question for each task in story.

        :tasks: list with length of tasks per story. Each entry is a string in
        the set {'tb','fb','sofb'}

        :questions: list with length of tasks per story. Each entry is a string
        in the set {'memory', 'reality', 'belief', 'search'}

        :statement_noise: probability of encountering noise sentence like 'The
        dog ran through the kitchen.'
        """

        idx_support_dummy = [0]
        actors = world.get_actors()
        locations = world.get_locations()
        objects = world.get_objects()
        containers = world.get_containers()

        random_actors = np.random.choice(actors, size=num_agents, replace=False)
        random_locations = np.random.choice(locations, size=num_locations, replace=False)
        random_objects = np.random.choice(objects, size=num_locations*2, replace=False)
        random_containers = np.random.choice(containers, size=num_locations*2, replace=False)

        oracle = Oracle(random_actors, random_locations, random_objects, random_containers)

        # Populate locations in the oracle with containers
        for i in range(len(random_locations)):
            location = random_locations[i]
            containers = random_containers[2*i:2*i+2]
            oracle.set_containers(location, list(containers))

        for i in range(len(random_objects)):
            oracle.set_object_container(random_objects[i], random_containers[i])

        start_state = oracle.locations.obj_containers.copy()

        chapters = {'fb':write_false_belief_chapter}

        story = []
        chapter = chapters[task]
        location = np.random.choice(random_locations)
        alt_loc = np.random.choice(random_locations)
        agent_ids = np.random.choice(
            range(len(random_actors)), size=3, replace=False
        )
        return chapter(
            start_state, oracle, location, agent_ids,
            random_actors, alternative_loc=alt_loc, question=question
        )