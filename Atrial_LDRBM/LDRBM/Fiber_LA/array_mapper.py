#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:55:02 2021

@author: Luca Azzolin

Copyright 2021 Luca Azzolin

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.  
"""
from carputils import tools

import Methods_LA as Method


def parser():
    # Generate the standard command line parser
    parser = tools.standard_parser()
    # Add arguments    
    parser.add_argument('--mesh1',
                        type=str,
                        default="",
                        help='path to meshname')
    parser.add_argument('--mesh2',
                        type=str,
                        default="",
                        help='path to meshname')
    parser.add_argument('--idat',
                        type=str,
                        default="",
                        help='input mesh format')
    parser.add_argument('--odat',
                        type=str,
                        default="",
                        help='input mesh format')
    parser.add_argument('--pts_or_cells',
                        default='points',
                        choices=['points',
                                 'cells'],
                        help='Mesh type')

    return parser


def jobID(args):
    ID = f"{args.mesh1.split('/')[-1]}_fibers"
    return ID


@tools.carpexample(parser, jobID)
def run(args, job):
    mesh1 = Method.smart_reader(args.mesh1)

    mesh2 = Method.smart_reader(args.mesh2)

    if args.pts_or_cells == "points":
        Method.point_array_mapper(mesh1, mesh2, args.mesh2, args.idat)
    else:
        Method.cell_array_mapper(mesh1, mesh2, args.mesh2, args.idat)


if __name__ == '__main__':
    run()
