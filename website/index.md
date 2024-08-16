---
title: TabularBench
datatable: true
---
{% include_relative style.css %}
<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css" />
<script src="https://code.jquery.com/jquery-3.5.1.js"></script>
<script src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>

<div class="top-menu">
    <a href="#" class="active"><i class="fas fa-home"></i> Leaderboard</a>
    <a href="https://github.com/serval-uni-lu/tabularbench" target="_blank"><i class="fab fa-github"></i> GitHub</a>
    <a href="https://serval-uni-lu.github.io/tabularbench/doc"><i class="fas fa-book"></i> Documentation</a>
</div>

<div class="content">

TabularBench: Adversarial robustness benchmark for tabular data

## Leaderboard

### CTU

{% include_relative ctu.md %}

### LCLD

{% include_relative lcld.md %}

### Malware

{% include_relative malware.md %}

### URL

{% include_relative url.md %}

### WIDS

{% include_relative wids.md %}

</div>

<script>
    $('table').DataTable({"bPaginate": false,})
</script>
