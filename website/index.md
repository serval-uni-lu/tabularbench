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

{% include_relative tables/ctu.md %}

### LCLD

{% include_relative tables/lcld.md %}

### Malware

{% include_relative tables/malware.md %}

### URL

{% include_relative tables/url.md %}

### WIDS

{% include_relative tables/wids.md %}

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
