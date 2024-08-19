---
title: TabularBench
datatable: true
---
<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css" />
<script src="https://code.jquery.com/jquery-3.5.1.js"></script>
<script src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>

TabularBench: Adversarial robustness benchmark for tabular data
[Documentation](https://serval-uni-lu.github.io/tabularbench/doc)

## MOEVA attack

You are currently viewing results for MOEVA attack. [View leaderboard](https://serval-uni-lu.github.io/tabularbench).

### CTU

{% include_relative moeva/ctu.md %}

### LCLD

{% include_relative moeva/lcld.md %}

### URL

{% include_relative moeva/url.md %}

### WIDS

{% include_relative moeva/wids.md %}

<script>
    $('table').DataTable({"bPaginate": false,})
</script>
