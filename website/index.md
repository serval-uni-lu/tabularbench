---
title: TabularBench
datatable: true
---
<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css" />
<script src="https://code.jquery.com/jquery-3.5.1.js"></script>
<script src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>

TabularBench: Adversarial robustness benchmark for tabular data
[Documentation](https://serval-uni-lu.github.io/tabularbench/doc)

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

<script>
    $('table').DataTable({"bPaginate": false,})
</script>
