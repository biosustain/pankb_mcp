"""
Resources: Provide static documentation and reference data for LLM context

Resources are for static/semi-static content like documentation, schemas, and reference data.
For dynamic data queries (genomes, genes, species), use Tools instead.

Right now OpenAI doesn't support resources, so we integrate those into system prompts defined on the client side. 
"""
from fastmcp import FastMCP

mcp = FastMCP(name="PanKBResources")


@mcp.resource("pankb://about")
def get_pankb_about() -> str:
    """Get PanKB project overview and introduction"""
    return """# PanKB - Pangenomic Knowledgebase

## Overview
PanKB is a pangenomic knowledgebase working to empower practitioners to leverage
microbial functions beyond those of a select few model organisms.

## Purpose
The platform enables researchers, biotechnologists, and strain engineers to access
and analyze microbial genomic data for applications in:
- Food production
- Human health
- Ecological sustainability

## Key Features

### Data Resources
- Growing dataset of pangenomic results across multiple microbial families
- Interactive data reports and analytics
- Global database search (genes, pathways, products, species)
- Alleleomes describing amino acid variants across gene alleles
- Dataset downloads for raw data and custom analysis
- Open-access pangenomic publication bibliography

### Pangenome Analytics
- Overview pages showing gene presence/absence matrices
- COG (Clusters of Orthologous Groups) distributions
- Gene annotation tables with detailed cluster information
- Phylogenetic trees with isolation source annotations

### AI-Powered Assistant
A specialized LLM chatbot focused on pangenomic literature, designed to provide
detailed answers with citations while avoiding hallucinated content.

## Data Applications
PanKB supports enzyme and strain engineering workflows:
- Gene identification for enzyme production
- Precise genetic modifications
- Pathway discovery and optimization
- Optimal starting strain selection

## Citation
Published in Nucleic Acids Research (2024)
Website: https://pankb.org

## Funding
Funded by the Novo Nordisk Foundation via the Center for Biosustainability
at the Technical University of Denmark.
"""


@mcp.resource("pankb://data-hierarchy")
def get_data_hierarchy() -> str:
    """Get explanation of PanKB data hierarchy and pangenomic concepts"""
    return """# PanKB Data Hierarchy

PanKB organizes genomic data in a hierarchical structure:

## Hierarchy Levels

```
Family (e.g., Bacillaceae)
  └── Species / Pangenome Analysis (e.g., Bacillus_subtilis)
        └── Genome (e.g., GCF_000009045.1)
              └── Gene (e.g., BSU_00010)
```

## Level Details

### 1. Family
- Highest taxonomic grouping in PanKB
- Contains multiple species
- Example: Enterobacteriaceae, Bacillaceae, Lactobacillaceae

### 2. Species (Pangenome Analysis)
- A pangenome analysis groups all genomes of a species
- Contains aggregated statistics: genome count, gene count, openness
- Gene classes distribution: Core, Shell, Cloud genes
- Named by species (e.g., "Escherichia_coli", "Bacillus_subtilis")

### 3. Genome
- Individual sequenced genome
- Has metadata: GC content, genome length, strain name
- Linked to isolation information: country, source (soil, blood, etc.)
- Assigned to phylogenetic groups (phylons)

### 4. Gene
- Individual gene within a genome/pangenome
- Annotated with:
  - Protein function description
  - Pangenomic class (Core/Accessory/Rare)
  - Frequency (how many genomes contain this gene)
  - COG category and name
  - KEGG pathway associations

## Pangenomic Gene Classes

| Class | Definition | Typical Frequency |
|-------|------------|-------------------|
| Core | Present in ≥95% of genomes | Essential genes |
| Shell (Accessory) | Present in 15-95% of genomes | Adaptive genes |
| Cloud (Rare) | Present in <15% of genomes | Strain-specific genes |

## Openness Score
A measure of pangenome openness (0-1):
- **Open pangenome** (high score): New genomes continue adding new genes
- **Closed pangenome** (low score): Most genes already discovered
"""


@mcp.resource("pankb://database-schema")
def get_database_schema() -> str:
    """Get PanKB database collections and field definitions"""
    return """# PanKB Database Schema

## MongoDB Collections

### 1. pankb_organisms
Pangenome analysis metadata for each species.

| Field | Type | Description |
|-------|------|-------------|
| family | string | Taxonomic family name |
| species | string | Species name |
| pangenome_analysis | string | Unique analysis identifier |
| genomes_num | int | Number of genomes in pangenome |
| genes_num | int | Total gene clusters |
| gene_class_distribution | array[3] | [core, shell, cloud] counts |
| openness | float | Pangenome openness score (0-1) |

### 2. pankb_genome_info
Individual genome metadata.

| Field | Type | Description |
|-------|------|-------------|
| genome_id | string | Unique genome identifier (e.g., GCF_xxx) |
| species | string | Species name |
| pangenome_analysis | string | Parent pangenome analysis |
| strain | string | Strain name |
| gc_content | float | GC content (0-1) |
| genome_len | int | Genome length in base pairs |
| phylo_group | string | Phylogenetic group assignment |

### 3. pankb_isolation_info
Genome isolation/environmental metadata.

| Field | Type | Description |
|-------|------|-------------|
| genome_id | string | Links to genome_info |
| country | string | Country of isolation |
| isolation_source | string | Source environment (Soil, Blood, etc.) |

### 4. pankb_gene_annotations
Gene functional annotations.

| Field | Type | Description |
|-------|------|-------------|
| gene | string | Gene cluster identifier |
| species | string | Species name |
| pangenome_analysis | string | Parent pangenome analysis |
| protein | string | Protein function description |
| pangenomic_class | string | Core / Accessory / Rare |
| frequency | float | Presence frequency across genomes |
| cog_category | string | COG category code (e.g., K, E, G) |
| cog_name | string | COG category full name |

### 5. pankb_gene_info
Detailed gene information for specific queries.

### 6. pankb_pathway_info
KEGG pathway database.

| Field | Type | Description |
|-------|------|-------------|
| pathway_id | string | KEGG pathway ID (e.g., map00010) |
| pathway_name | string | Pathway name |

### 7. pankb_genome_phylons
Phylogenetic group assignments for genomes.

### 8. pankb_gene_phylons
Phylogenetic group assignments for genes.

### 9. pankb_stats
Database statistics snapshots.

| Field | Type | Description |
|-------|------|-------------|
| date | datetime | Snapshot date |
| pankb_dimensions | object | Overall counts |
| organism_genome_count | object | Genomes per family |
| organism_gene_count | object | Genes per family |
| country_strain_count | object | Strains per country |

## Azure Cosmos DB (Vector Store)
RAG knowledge base for pangenomic literature.

| Field | Type | Description |
|-------|------|-------------|
| vectorContent | vector | Document embeddings (VoyageAI) |
| textContent | string | Document text |
| source | string | Source URL |
| title | string | Document title |

## Azure Blob Storage
Large genomic data files (JSON, gzipped JSON).

Base URL: `https://pankb.blob.core.windows.net/data/PanKB/web_data_v2/`
Structure: `species/{species_name}/{filename}.json.gz`
"""


@mcp.resource("pankb://cog-categories")
def get_cog_categories() -> str:
    """Get COG (Clusters of Orthologous Groups) category reference"""
    return """# COG Categories Reference

COG (Clusters of Orthologous Groups) categories classify genes by function.

## Information Storage and Processing

| Code | Category |
|------|----------|
| J | Translation, ribosomal structure and biogenesis |
| A | RNA processing and modification |
| K | Transcription |
| L | Replication, recombination and repair |
| B | Chromatin structure and dynamics |

## Cellular Processes and Signaling

| Code | Category |
|------|----------|
| D | Cell cycle control, cell division, chromosome partitioning |
| Y | Nuclear structure |
| V | Defense mechanisms |
| T | Signal transduction mechanisms |
| M | Cell wall/membrane/envelope biogenesis |
| N | Cell motility |
| Z | Cytoskeleton |
| W | Extracellular structures |
| U | Intracellular trafficking, secretion, and vesicular transport |
| O | Posttranslational modification, protein turnover, chaperones |

## Metabolism

| Code | Category |
|------|----------|
| C | Energy production and conversion |
| G | Carbohydrate transport and metabolism |
| E | Amino acid transport and metabolism |
| F | Nucleotide transport and metabolism |
| H | Coenzyme transport and metabolism |
| I | Lipid transport and metabolism |
| P | Inorganic ion transport and metabolism |
| Q | Secondary metabolites biosynthesis, transport and catabolism |

## Poorly Characterized

| Code | Category |
|------|----------|
| R | General function prediction only |
| S | Function unknown |

## Not in COG

| Code | Category |
|------|----------|
| - | Not assigned to any COG category |
"""
