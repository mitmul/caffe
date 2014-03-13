#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin cifar10_80sec_solver.prototxt

#reduce learning rate by fctor of 10 after 8 epochs
GLOG_logtostderr=1 $TOOLS/train_net.bin cifar10_80sec_solver_lr1.prototxt cifar10_80sec_iter_4000.solverstate
