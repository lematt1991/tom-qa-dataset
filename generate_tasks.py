import argparse
import logging
import glob
import numpy as np
import os
import sys
import random
import itertools

from stringify import stringify
from tasks import Specify_Tasks
from tasks_v2 import Specify_Tasks as Specify_Tasks_v2
from utils import is_file, mkdir_p, remove_extension
from world import World
from tqdm import tqdm
from subprocess import check_output


def generate_tasks_with_oracle_fixed_count(
    world_paths, output_dir_path, n, noise=.1, train_noise=False
):
    """Generates stories with guarantee that each task is seen n times."""

    mkdir_p(output_dir_path)
    n = n[0]

    for world in world_paths:

        # Load information from world
        w = World()
        w.load(world)
        world_name = remove_extension(world)

        # ----------------------------- TRAINING ----------------------------- #

        # Create folder to contain data
        mkdir_p(output_dir_path)
        train_file_path = os.path.join(
            output_dir_path, 'qa21_task_AB_train.txt'
        )

        # Define task creator and task types
        task = Specify_Tasks()
        tasks = ['tb', 'fb', 'sofb']
        questions = ['memory', 'reality', 'search', 'belief']

        combo = itertools.product(tasks, questions, ['val', 'test', 'train'])
        for task_type, question, data_set in combo:
            fname = '%s_%s_%s.txt' % (task_type, question, data_set)
            path = os.path.join(output_dir_path, fname)

            with open(path, 'w') as f:
                stories = []
                for i in tqdm(range(n)):
                    story = task.generate_story(
                        w, 1, [task_type], [question], num_agents=4,
                        num_locations=6, statement_noise=0 if not train_noise
                            and data_set == 'train' else noise
                    )
                    f.write('\n'.join(stringify(story)))
                    f.write('\n')

def generate_tasks_with_oracle_fixed_count_1_task_1_story(
    world_paths, output_dir_path, n, noise=.1, train_noise=False
):
    """Generates stories with guarantee that each task is seen n times."""
    mkdir_p(output_dir_path)
    n = n[0]

    for world in world_paths:

        w = World()
        w.load(world)
        world_name = remove_extension(world)

        # ----------------------------- TRAINING ----------------------------- #

        # Create folder to contain data
        mkdir_p(output_dir_path)

        # Define task creator and task types
        task = Specify_Tasks()
        tasks = ['tb', 'fb', 'sofb']
        questions = ['memory', 'reality', 'search', 'belief']

        combo = itertools.product(tasks, questions, ['val', 'test', 'train'])
        for task_type, question, data_set in combo:
            fname = '%s_%s_%s.txt' % (task_type, question, data_set)
            path = os.path.join(output_dir_path, fname)

            with open(path, 'w') as f:
                stories = []
                for i in tqdm(range(n)):
                    story = task.generate_story(
                        w, 1, [task_type], [question], num_agents=4,
                        num_locations=6, statement_noise=0 if not train_noise
                            and data_set == 'train' else noise
                    )
                    f.write('\n'.join(stringify(story)))
                    f.write('\n')


def generate_tasks_v2(
    world_paths, output_dir_path, n, noise=.1, train_noise=False,
    all_questions=False
):
    """Generates stories with guarantee that each task is seen n times."""
    mkdir_p(output_dir_path)
    n = n[0]

    for world in world_paths:

        w = World()
        w.load(world)
        world_name = remove_extension(world)

        # ----------------------------- TRAINING ----------------------------- #

        # Create folder to contain data
        mkdir_p(output_dir_path)

        # Define task creator and task types
        task = Specify_Tasks_v2()
        tasks = ['fb']
        questions = ['all'] if all_questions else ['memory', 'reality', 'search', 'belief']

        # ---------------------------- VAL + TEST + train ---------------------------- #
        # Iterate through all testing conditions
        combo = itertools.product(tasks, questions, ['val', 'test', 'train'])
        for task_type, question, data_set in combo:
            fname = '%s_%s_%s.txt' % (task_type, question, data_set)
            tname = '%s_%s_%s.trace' % (task_type, question, data_set)
            path = os.path.join(output_dir_path, fname)
            traces = os.path.join(output_dir_path, tname)
            with open(path, 'w') as f, open(traces, 'w') as trace_f:
                stories = []
                for i in tqdm(range(n)):
                    res = task.generate_story(
                        w, task_type, question, num_agents=4,
                        num_locations=6, statement_noise=0 if not train_noise
                            and data_set == 'train' else noise
                    )
                    for story, trace in zip(*res):
                        print('\n'.join(stringify(story)), file=f)
                        print(','.join(trace), file=trace_f)
    
        outfile = os.path.join(output_dir_path, 'train_all.txt')
        check_output(f'cat {os.path.join(output_dir_path, "*_train.txt")} > {outfile}', shell=True)


def parse_args(args):

    parser = argparse.ArgumentParser(
        description='Process command-line arguments.'
    )

    parser.add_argument(
        '-w', '--world_path', dest='world_paths', type=is_file, required=True,
        action='append', help='Path to a world definition file'
    )

    parser.add_argument(
        '-o', '--output_dir_path', dest='output_dir_path', type=mkdir_p,
        default='data', help='Output directory path'
    )

    parser.add_argument(
        '-l', '--logging', type=str, default='INFO', metavar='logging',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level'
    )

    parser.add_argument(
        '-n', '--num_stories', dest='num_stories_choices', type=int,
        action='append', required=True,
        help='Number of stories (examples) in a task)'
    )

    parser.add_argument(
        '-easy', '--easy', dest='easy', action='store_true',
        help='Switch on tom-easy generation'
    )

    parser.add_argument(
        '-ptn', '--prob_test_noise', dest='test_noise', type=float,
        required=True, help='Probability of encountering random noise sentence'
    )

    parser.add_argument(
        '--all_questions', action='store_true', help='Create all possible questions'
        ' for each story'
    )

    parser.add_argument(
        '-tn', '--train_noise', dest='train_noise', type=bool, default=False,
        help='Whether or not to include noise at training time'
    )

    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    parser.add_argument('-v', '--version', choices=['v1', 'v2'], default='v1')

    parsed = parser.parse_args(args)

    return parsed


def main(args=sys.argv[1:]):

    args = parse_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(
        level=args.logging, format='%(asctime)s\t%(levelname)-8s\t%(message)s'
    )

    if args.version == 'v2':
        generate_tasks_v2(
            world_paths=args.world_paths,
            output_dir_path=args.output_dir_path,
            n=args.num_stories_choices,
            noise=args.test_noise,
            train_noise=args.train_noise,
            all_questions=args.all_questions,
        )
    elif args.easy:
        generate_tasks_with_oracle_fixed_count_1_task_1_story(
            world_paths=args.world_paths,
            output_dir_path=args.output_dir_path,
            n=args.num_stories_choices,
            noise=args.test_noise,
            train_noise=args.train_noise
        )
    else:
        generate_tasks_with_oracle_fixed_count(
            world_paths=args.world_paths,
            output_dir_path=args.output_dir_path,
            n=args.num_stories_choices,
            noise=args.test_noise,
            train_noise=args.train_noise
        )


if __name__ == "__main__":
    sys.exit(main())
