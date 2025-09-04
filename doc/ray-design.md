# Parallel in LLMEngine

## Case: no parallelism

Run normally as a single process.

## Case: parallelism

LLMEngine -> List of stages -> List of workers

List of stages maintains the sheduler of the stages.