---
title: TabularBench
datatable: true
---
<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css" />
<script src="https://code.jquery.com/jquery-3.5.1.js"></script>
<script src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>

TabularBench: Adversarial robustness benchmark for tabular data

## Leaderboard

### CTU

{% include_relative ctu.html %}

### LCLD

{% include_relative lcld.html %}

### Malware

{% include_relative malware.html %}

### URL

{% include_relative url.html %}

### WIDS

{% include_relative wids.html %}

<script>
    $('table').DataTable({"bPaginate": false,})
</script>
