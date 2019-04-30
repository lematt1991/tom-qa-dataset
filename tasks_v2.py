import numpy as np


from clause import Clause, Question
from oracle import Oracle
from dynamic_actions import *
from collections import defaultdict
import random

def sample_question(oracle_start_state, oracle, agent1, agent2, obj, question):
    idx_dummy = [0]
    questions = [Question(idx_dummy, SearchedAction(oracle, agent1, obj)),
                 Question(idx_dummy, SearchedAction(oracle, agent2, obj)),
                 Question(idx_dummy, BeliefSearchAction(oracle, agent1, agent2, obj)),
                 Question(idx_dummy, RealityAction(oracle, obj)),
                 Question(idx_dummy, MemoryAction(oracle_start_state, obj))
                ]
    if question:
        if question == 'memory':
            return questions[-1]
        elif question == 'reality':
            return questions[3]
        elif question == 'belief':
            return questions[2]
        elif question == 'search':
            return questions[1]
    return np.random.choice(questions)

# -------------------------------- Chapters ---------------------------------- #

def write_true_belief_chapter(
    start_state, oracle, location, agent_ids, all_agents, questions=None
):
    """
    Creates list of clauses that constitute
    a true belief task.

    agent_ids: list that gives indices of agents
      in container. should be length 2.
    all_agents: list of all agents
    container: container to which the object is
      moved
    question: one of ['memory', 'reality', 'belief', 'search', None]
      if None, then pick randomly

    Warning: clauses will advance that state
    of the simulation, should clauses should
    be appended in order.
    """
    a1, a2 = all_agents[agent_ids[0]], all_agents[agent_ids[1]]
    agent_ids = [aid+1 for aid in agent_ids]

    # Pick random object at location
    obj = np.random.choice(oracle.get_objects_at_location(location))
    container_1 = oracle.get_object_container(obj)

    # Pick random container in locations
    container_candidates = oracle.get_containers(location)[:]
    container_candidates.remove(container_1)
    container_2 = np.random.choice(container_candidates)

    chapter = []

    # Move agents into location
    if oracle.get_location(a1) == location:
        chapter.extend([
            Clause([agent_ids[0]], LocationAction(oracle, (a1, location)))
        ])
    else:
        chapter.extend([
            Clause([agent_ids[0]], EnterAction(oracle, (a1, location)))
        ])

    if oracle.get_location(a2) == location:
        chapter.extend([
            Clause(agent_ids, LocationAction(oracle, (a2, location)))
        ])
    else:
        chapter.extend([
            Clause(agent_ids, EnterAction(oracle, (a2, location), [a1]))
        ])

    chapter.extend([
        Clause(agent_ids, ObjectLocAction(oracle, obj, [a1, a2])),
        Clause(agent_ids, MoveAction(oracle, (a1, obj, container_2), [a2])),
    ])

    for question in questions:
        chapter.append(
            sample_question(start_state, oracle, a1, a2, obj, question)
        )

    return chapter

class Agent:
    def __init__(self, id : int, name : str, order : int):
        self.id = id + 1  # these are 1 indexed for some reason...
        self.name = name
        self.order = order

def ids(agents):
    return [agent.id for agent in agents]

def names(agents):
    return [agent.name for agent in agents]

def write_false_belief_chapter(
    start_state, oracle, location, agent_ids, all_agents,
    alternative_loc, questions=None
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
    a1, a2 = tuple(Agent(id, all_agents[id], i) for i, id in enumerate(agent_ids))

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
        if oracle.get_location(agent.name) == location:
            action = LocationAction(oracle, (agent.name, location), names(enter_observers))
        else:
            action = EnterAction(oracle, (agent.name, location), names(enter_observers))
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
            trace.append(f'agent_{a2.order}_reenters')

    # JUST ONE QUESTION SPLITS A STRING TODO TODO
    for question in questions:
        agents = [a1, a2]
        random.shuffle(agents)
        if question in {'memory', 'reality'}:
            trace.append(question)
        elif question == 'search':
            trace.append(f'first_order_agent_{agents[1].order}')
        else:
            trace.append(f'second_order_agent{agents[0].order}')
        chapter.append(sample_question(start_state, oracle, agents[0].name,
            agents[1].name, obj, question))

    return chapter, ','.join(trace)

def write_second_order_false_belief_chapter(
    start_state, oracle, location, agent_ids, all_agents, questions=None
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
    a1, a2 = all_agents[agent_ids[0]], all_agents[agent_ids[1]]
    agent_ids = [aid+1 for aid in agent_ids]

    # pick random object at location
    obj = np.random.choice(oracle.get_objects_at_location(location))
    container_1 = oracle.get_object_container(obj)

    # pick random container in locations
    container_candidates = oracle.get_containers(location)[:]
    container_candidates.remove(container_1)
    container_2 = np.random.choice(container_candidates) # set would be more elegant

    chapter = []

    # move agents into location
    if oracle.get_location(a1) == location:
        chapter.extend([Clause([agent_ids[0]], LocationAction(oracle, (a1, location)))])
    else:
        chapter.extend([Clause([agent_ids[0]], EnterAction(oracle, (a1, location)))])

    if oracle.get_location(a2) == location:
        chapter.extend([Clause(agent_ids, LocationAction(oracle, (a2, location)))])
    else:
        chapter.extend([Clause(agent_ids, EnterAction(oracle, (a2, location), [a1]))])

    chapter.extend([
        Clause(agent_ids, ObjectLocAction(oracle, obj, [a1, a2])),
        Clause(agent_ids, ExitedAction(oracle, (a2))),
        Clause([agent_ids[0]], MoveAction(oracle, (a1, obj, container_2))),
        Clause([agent_ids[0]], ExitedAction(oracle, (a1))),
        Clause([agent_ids[1]], EnterAction(oracle, (a2, location))),
        #Clause([agent_ids[1]], PeekAction(oracle, (a2, container_2))), # closed container condition
        #Clause([agent_ids[0]], EnterAction(oracle, (a1, location), [a2])), # closed container condition
        # sample_question(start_state, oracle, a1, a2, obj, question)
    ])

    for question in questions:
        chapter.append(sample_question(start_state, oracle, a1, a2, obj, question))

    return chapter

#######################################
############### Tasks #################
#######################################

class Task(object):

    def __init__(self,
                 num_questions=5,
                 exit_prob=1.,
                 informant_prob=1.,
                 search_prob=1.,
                 test_cond='first order'):

        self.num_questions = num_questions

        self.search_prob = search_prob

        self.exit_inform_probs = [1 - exit_prob,
                                  exit_prob * (1 - informant_prob),
                                  exit_prob * informant_prob]
        assert sum(self.exit_inform_probs) == 1

        assert test_cond in ['first order',
                             'second order',
                             'reality',
                             'memory'], \
            "Invalid test condition: %s" % test_cond
        self.test_cond = test_cond

    def generate_story(self, world):
        raise NotImplementedError("Abstract method.")

class Specify_Tasks(Task):

    def generate_story(
        self, world, tasks_per_story, tasks, questions, num_agents=6,
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

        chapters = {'tb':write_true_belief_chapter, 'fb':write_false_belief_chapter,
                    'sofb':write_second_order_false_belief_chapter}

        story = []
        for i in range(tasks_per_story):
            chapter = chapters[tasks[i]]
            location = np.random.choice(random_locations)
            alt_loc = np.random.choice(random_locations)
            agent_ids = np.random.choice(
                range(len(random_actors)), size=2, replace=False
            )
            story.extend(
                chapter(
                    start_state, oracle, location, agent_ids,
                    random_actors, alternative_loc=alt_loc, questions=[questions[i]]
                )
            )

        if statement_noise:
            noisy_story = []
            prev_i = 0
            noise = [i for i in range(len(story)) if np.random.rand() < statement_noise]
            for i in noise:
                noisy_story.extend(story[prev_i:i] + [Clause([], NoiseAction())])
                prev_i = i
            noisy_story.extend(story[prev_i:])

            return noisy_story

        return story

    def generate_story_qs_at_end(
        self, world, tasks_per_story, tasks, questions, num_agents=6,
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

        # Fetch agents and objects and select a random subset
        idx_support_dummy = [0]
        actors = world.get_actors()
        locations = world.get_locations()
        objects = world.get_objects()
        containers = world.get_containers()

        random_actors = np.random.choice(
            actors, size=num_agents, replace=False
        )
        random_locations = np.random.choice(
            locations, size=num_locations, replace=False
        )
        random_objects = np.random.choice(
            objects, size=num_locations*2, replace=False
        )
        random_containers = np.random.choice(
            containers, size=num_locations*2, replace=False
        )

        # Create the oracle
        oracle = Oracle(
            random_actors, random_locations, random_objects, random_containers
        )

        # Populate locations in the oracle with containers
        for i in range(len(random_locations)):
            location = random_locations[i]
            containers = random_containers[2*i:2*i+2]
            oracle.set_containers(location, list(containers))

        # Populate containers with objects
        for i in range(len(random_objects)):
            oracle.set_object_container(random_objects[i], random_containers[i])

        # Need start state for memory question
        start_state = oracle.locations.obj_containers.copy()

        # Create story by task
        chapters = {'tb':write_true_belief_chapter,
                    'fb':write_false_belief_chapter,
                    'sofb':write_second_order_false_belief_chapter}
        story = []
        for i in range(tasks_per_story-1):
            chapter = chapters[tasks[i]]
            location = np.random.choice(random_locations)
            agent_ids = np.random.choice(
                range(len(random_actors)), size=2, replace=False
            )
            story.extend(
                chapter(
                    start_state, oracle, location, agent_ids, random_actors, []
                )
            )
        chapter = chapters[tasks[-1]]
        location = np.random.choice(random_locations)
        agent_ids = np.random.choice(
            range(len(random_actors)), size=2, replace=False
        )
        story.extend(
            chapter(
                start_state, oracle, location, agent_ids, random_actors, questions
            )
        )

        # At the end, at noise sentences randomly
        if statement_noise:
            noisy_story = []
            prev_i = 0
            noise = [i for i
                in range(len(story)) if np.random.rand() < statement_noise
            ]
            for i in noise:
                noisy_story.extend(
                    story[prev_i:i] + [Clause([], NoiseAction())]
                )
                prev_i = i
            noisy_story.extend(story[prev_i:])

            return noisy_story

        return story