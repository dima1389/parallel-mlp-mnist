# PlantUML Diagram Atlas — parallel-mlp-mnist

Comprehensive visual documentation for the **parallel-mlp-mnist** project:
a C++ multi-layer perceptron trained on MNIST with three parallelism strategies
(sequential, OpenMP, MPI).

---

## Rendering Instructions

Any PlantUML tool can render `.puml` files:

- **VS Code**: Install the [PlantUML extension](https://marketplace.visualstudio.com/items?itemName=jebbs.plantuml) (requires Java + Graphviz).
- **Online**: Paste source at <https://www.plantuml.com/plantuml/uml/>.
- **CLI**: `plantuml docs/diagrams/**/*.puml` (produces PNG beside each file).
- **Docker**: `docker run -v "$PWD":/data plantuml/plantuml -tsvg /data/docs/diagrams/**/*.puml`

---

## Diagram Inventory

| ID | Title | Type | Category | Purpose | Confidence |
|----|-------|------|----------|---------|------------|
| D01 | System Context | Context | 01-context | Shows the system in its environment: user, MNIST data, Docker Hub, GitHub Actions | High |
| D02 | High-Level Architecture | Package | 02-architecture | Top-level view of entry points, core library, data, tooling, and infrastructure | High |
| D03 | Layered Architecture | Rectangle | 02-architecture | Five-layer decomposition: entry points → MLP → math primitives → I/O → OS/runtime | High |
| D04 | Component Overview | Component | 03-components | C4-style component diagram showing all modules and their dependency edges | High |
| D05 | Module Include Dependencies | Dependency | 03-components | Header-level `#include` graph for all source and header files | High |
| D06 | Core Data Structures | Class | 04-domain-model | UML class diagram of `Layer`, `MLP`, `Dataset`, `Config`, `Timer`, `EpochRecord` | High |
| D07 | Layer Memory Model | Object | 04-domain-model | Per-layer heap allocation layout for forward and backward pass buffers | High |
| D08 | MNIST Data Model | Class | 05-data | IDX binary format structure and in-memory `Dataset` representation | High |
| D09 | MLP Forward Pass | Sequence | 06-sequences | Step-by-step forward propagation for a single sample | High |
| D10 | MLP Backward Pass | Sequence | 06-sequences | Backpropagation: output delta, hidden delta propagation, gradient accumulation | High |
| D11 | Sequential Training Loop | Sequence | 06-sequences | Full epoch sequence for the single-threaded training binary | High |
| D12 | OpenMP Gradient Aggregation | Sequence | 06-sequences | Per-mini-batch parallel forward/backward and sequential gradient merge | High |
| D13 | MPI Gradient Allreduce | Sequence | 06-sequences | Per-mini-batch data-parallel processing and `MPI_Allreduce` gradient sync | High |
| D14 | Sequential Training Flow | Activity | 07-activities | Activity diagram for `train_sequential.cpp` from CLI parse to cleanup | High |
| D15 | OpenMP Training Flow | Activity (swimlane) | 07-activities | Swimlane showing main thread vs worker threads across the training loop | High |
| D16 | MPI Training Flow | Activity (swimlane) | 07-activities | Swimlane for rank 0 vs all ranks: data loading, batch sharding, Allreduce, reporting | High |
| D17 | Benchmark Pipeline | Activity | 07-activities | `benchmark.sh` flow: nested loops over configs, thread/process counts, and runs | High |
| D18 | Training Job States | State | 08-states | State machine for a single binary invocation: init → training → saving → cleanup | High |
| D19 | Docker Multi-Stage Build | Deployment | 11-deployment | Builder stage (compile) → runtime stage (slim image) copy and dependency flow | High |
| D20 | Docker Compose Services | Deployment | 11-deployment | All Compose services, named volumes, bind mounts, and `depends_on` relationships | High |
| D21 | GitHub Actions CI Pipeline | Activity | 13-cicd | Four-step pipeline: checkout → Docker build → unit tests → benchmark → artifact upload | High |
| D22 | Build Targets | Dependency | 15-dependencies | Makefile target graph: `all`, `seq`, `omp`, `mpi`, `test`, `debug` and their object files | High |

---

## File Structure

```
docs/diagrams/
├── index.md                          ← this file
├── 01-context/
│   └── D01-system-context.puml
├── 02-architecture/
│   ├── D02-high-level-architecture.puml
│   └── D03-layered-architecture.puml
├── 03-components/
│   ├── D04-component-overview.puml
│   └── D05-module-dependencies.puml
├── 04-domain-model/
│   ├── D06-data-structures.puml
│   └── D07-layer-memory-model.puml
├── 05-data/
│   └── D08-mnist-data-model.puml
├── 06-sequences/
│   ├── D09-forward-pass.puml
│   ├── D10-backward-pass.puml
│   ├── D11-sequential-training.puml
│   ├── D12-openmp-gradient-aggregation.puml
│   └── D13-mpi-gradient-allreduce.puml
├── 07-activities/
│   ├── D14-sequential-training-flow.puml
│   ├── D15-openmp-training-flow.puml
│   ├── D16-mpi-training-flow.puml
│   └── D17-benchmark-flow.puml
├── 08-states/
│   └── D18-training-job-states.puml
├── 11-deployment/
│   ├── D19-docker-multistage-build.puml
│   └── D20-docker-compose-services.puml
├── 13-cicd/
│   └── D21-github-actions-pipeline.puml
└── 15-dependencies/
    └── D22-build-targets.puml
```

---

## Cross-Reference Map

| To understand… | Start with | Then see |
|----------------|-----------|---------|
| What the system does | D01 | D02, D03 |
| How modules relate | D04 | D05 |
| The neural network internals | D06 | D07, D09, D10 |
| MNIST data loading | D08 | D09 |
| Sequential training | D11 | D14, D18 |
| OpenMP parallelism | D12 | D15 |
| MPI parallelism | D13 | D16 |
| Benchmarking workflow | D17 | D21 |
| Docker / containers | D19 | D20 |
| CI pipeline | D21 | D19, D17 |
| Build system | D22 | D04 |

---

## Evidence and Confidence

All diagrams are derived exclusively from confirmed project sources:

- **Source code** (`src/*.cpp`, `include/*.h`) — structural, algorithmic, and data-flow diagrams
- **Makefile** — build target and compilation flag diagrams  
- **Dockerfile + docker-compose.yml** — deployment diagrams  
- **.github/workflows/ci.yml** — CI/CD pipeline diagram  
- **scripts/benchmark.sh** — benchmark flow diagram  
- **configs/*.conf** — network configuration details  
- **README.md + docs/\*** — supplementary context  

No components, protocols, or infrastructure elements were invented.
