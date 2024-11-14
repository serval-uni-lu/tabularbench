---
datatable: true
---
<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css" />
<script src="https://code.jquery.com/jquery-3.5.1.js"></script>
<script src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>
<link rel="stylesheet" href="{{ '/assets/css/custom.css' | relative_url }}">

TabularBench: Adversarial robustness benchmark for tabular data.

[Documentation](https://serval-uni-lu.github.io/tabularbench/doc).

## Leaderboard

You are currently viewing results of the leaderboard. View MOEVA attack results [here](moeva).

Jump to dataset:

- [CTU](#ctu)
- [LCLD](#lcld)
- [Malware](#malware)
- [URL](#url)
- [WIDS](#wids)

### CTU

{% include_relative data/ctu.md %}

### LCLD

{% include_relative data/lcld.md %}

### Malware

{% include_relative data/malware.md %}

### URL

{% include_relative data/url.md %}

### WIDS

{% include_relative data/wids.md %}

<script>
    var table = $('table').DataTable(
        {
            "bPaginate": false,
            "language": {
                searchPlaceholder: 'Architectures, training methods, etc.'
            },
            // "autoWidth": true,
        }
    );
    table.columns.adjust().draw();
</script>
